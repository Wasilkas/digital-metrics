"""Significance tests for per-class metric deltas between a baseline and a candidate model.

Every delta is ``candidate - baseline``, so a positive delta is an improvement and a
negative one a degradation. Every test is two-sided: the p-value answers "did this class
change at all", the sign of ``delta`` answers "in which direction". Benjamini-Hochberg
then controls the false discovery rate over that two-sided family, and the direction of
each surviving change is still read off the sign of its delta.

One hypothesis gets exactly one p-value. Recall's comes from :func:`mcnemar_recall` and
precision's from :func:`bootstrap_precision_delta`; :func:`bootstrap_recall_delta`
supplies recall's confidence interval only and deliberately returns a NaN p-value, so
the recall hypothesis cannot be counted twice in the Benjamini-Hochberg family.
"""

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import binomtest, false_discovery_control


class TestResult(BaseModel):
    """The observed delta, its p-value, its interval and its sample sizes.

    ``ci_lower``/``ci_upper`` are ``None`` when the producer yields no interval
    (McNemar) or when no resample was usable. ``p_value`` is NaN when the producer
    yields no p-value (the recall bootstrap) or when the test is undefined.

    ``n_baseline``/``n_candidate`` count aligned ground-truth boxes for McNemar,
    prediction rows for the precision bootstrap and ground-truth rows for the recall
    bootstrap.
    """

    delta: float
    p_value: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    n_baseline: int
    n_candidate: int


class BHResult(BaseModel):
    """Benjamini-Hochberg output, aligned element-wise to the input p-values."""

    adjusted_p_values: list[float]
    rejected: list[bool]
    q: float = 0.05


def mcnemar_recall(
    baseline_detected: pd.Series,
    candidate_detected: pd.Series,
    *,
    diagnostic: Callable[[str], None] | None = None,
) -> TestResult:
    """Exact two-sided McNemar test on the boxes both models were scored against.

    Recall is paired at the ground-truth-box level, so each box contributes one
    detected/missed outcome per model. The exact binomial test is used instead of the
    chi-square approximation because per-class discordant counts are small, and its
    ``alternative`` is pinned so a lost recall is flagged as loudly as a gained one.
    """
    shared = baseline_detected.index.intersection(candidate_detected.index)
    if len(shared) == 0:
        if diagnostic is not None:
            diagnostic("McNemar test got no overlapping ground-truth boxes")
        return TestResult(delta=float("nan"), p_value=float("nan"), n_baseline=0, n_candidate=0)

    baseline = baseline_detected.loc[shared]
    candidate = candidate_detected.loc[shared]
    if baseline.isna().any() or candidate.isna().any():
        # astype(bool) maps a missing flag to True, which would silently inflate recall.
        raise ValueError("Detection flags must not contain missing values")

    baseline = baseline.astype(bool)
    candidate = candidate.astype(bool)
    baseline_only = int((baseline & ~candidate).sum())
    candidate_only = int((~baseline & candidate).sum())
    discordant = baseline_only + candidate_only

    # No discordant pair carries no evidence in either direction.
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(baseline_only, discordant, 0.5, alternative="two-sided").pvalue)
    return TestResult(
        delta=float(candidate.mean() - baseline.mean()),
        p_value=p_value,
        n_baseline=len(shared),
        n_candidate=len(shared),
    )


def _per_image_counts(
    status: pd.DataFrame,
    images: Sequence[str],
    flag_column: str,
    description: str,
    diagnostic: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Successes and row counts per image, in the order of ``images``."""
    split_images = pd.Index(images)
    grouped = status.groupby("image_name")[flag_column]
    successes = grouped.sum().reindex(split_images, fill_value=0).to_numpy(dtype=float)
    totals = grouped.size().reindex(split_images, fill_value=0).to_numpy(dtype=float)
    dropped = len(status) - int(totals.sum())
    if dropped and diagnostic is not None:
        diagnostic(
            f"{dropped} {description} rows reference images outside the split and were dropped"
        )
    return successes, totals


def _bootstrap_rate_delta(
    baseline_status: pd.DataFrame,
    candidate_status: pd.DataFrame,
    images: Sequence[str],
    *,
    flag_column: str,
    metric: str,
    iterations: int,
    seed: int,
    diagnostic: Callable[[str], None] | None,
) -> TestResult:
    """Resample images to bootstrap the candidate-minus-baseline delta of a pooled rate.

    Both precision and recall are a success count over a row count that differs between
    the two models, so neither pairs at row level and both share this machinery. Images
    are resampled rather than rows because boxes cluster within images and resampling
    rows would understate the variance. A draw in which either model contributes no row
    leaves its rate undefined and is skipped.

    The p-value compares absolute deviations against the H0-centred distribution, which
    makes it two-sided - a drop is as significant as an equal-sized gain. Do not narrow
    that comparison to one tail. It averages over the usable draws only, but its floor
    stays ``1 / iterations`` because that is the resolution the bootstrap was asked for.
    """
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if len(images) == 0:
        raise ValueError("images must not be empty")

    baseline_hits, baseline_totals = _per_image_counts(
        baseline_status, images, flag_column, f"baseline {metric}", diagnostic
    )
    candidate_hits, candidate_totals = _per_image_counts(
        candidate_status, images, flag_column, f"candidate {metric}", diagnostic
    )
    n_baseline = int(baseline_totals.sum())
    n_candidate = int(candidate_totals.sum())
    if n_baseline == 0 or n_candidate == 0:
        if diagnostic is not None:
            diagnostic(
                f"The {metric} bootstrap needs rows from both models, "
                f"got {n_baseline} and {n_candidate}"
            )
        return TestResult(
            delta=float("nan"),
            p_value=float("nan"),
            n_baseline=n_baseline,
            n_candidate=n_candidate,
        )

    observed = candidate_hits.sum() / n_candidate - baseline_hits.sum() / n_baseline

    rng = np.random.default_rng(seed)
    image_count = len(images)
    resampled = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.integers(0, image_count, image_count)
        baseline_rows = baseline_totals[sampled].sum()
        candidate_rows = candidate_totals[sampled].sum()
        if baseline_rows == 0 or candidate_rows == 0:
            resampled[iteration] = np.nan
            continue
        resampled[iteration] = (
            candidate_hits[sampled].sum() / candidate_rows
            - baseline_hits[sampled].sum() / baseline_rows
        )

    usable = resampled[np.isfinite(resampled)]
    if usable.size == 0:
        if diagnostic is not None:
            diagnostic(f"Every {metric} bootstrap draw was degenerate; the interval is undefined")
        return TestResult(
            delta=float(observed),
            p_value=float("nan"),
            n_baseline=n_baseline,
            n_candidate=n_candidate,
        )

    ci_lower, ci_upper = np.percentile(usable, [2.5, 97.5])
    # The interval comes from the uncentred draws, the p-value from the H0-centred ones.
    centred = usable - observed
    exceedance = float(np.mean(np.abs(centred) >= abs(observed)))
    return TestResult(
        delta=float(observed),
        p_value=max(exceedance, 1.0 / iterations),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        n_baseline=n_baseline,
        n_candidate=n_candidate,
    )


def bootstrap_precision_delta(
    baseline_pred_status: pd.DataFrame,
    candidate_pred_status: pd.DataFrame,
    images: Sequence[str],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    diagnostic: Callable[[str], None] | None = None,
) -> TestResult:
    """Bootstrap the precision delta from ``pred_index, image_name, instance_label, is_tp``.

    This is the authoritative precision test: it supplies both the interval and the
    p-value, because precision has no exact paired counterpart the way recall has
    McNemar.
    """
    return _bootstrap_rate_delta(
        baseline_pred_status,
        candidate_pred_status,
        images,
        flag_column="is_tp",
        metric="precision",
        iterations=iterations,
        seed=seed,
        diagnostic=diagnostic,
    )


def bootstrap_recall_delta(
    baseline_gt_status: pd.DataFrame,
    candidate_gt_status: pd.DataFrame,
    images: Sequence[str],
    *,
    iterations: int = 10_000,
    seed: int = 0,
    diagnostic: Callable[[str], None] | None = None,
) -> TestResult:
    """Bootstrap a recall-delta interval from ``gt_index, image_name, instance_label, detected``.

    This contributes the INTERVAL ONLY. ``p_value`` is always NaN by design: the
    authoritative recall p-value is :func:`mcnemar_recall`, which is exact and stays
    trustworthy at the per-class counts this report hits (a rare class can have well
    under twenty boxes in a split, where a bootstrap p-value is unreliable). Returning a
    second recall p-value here would invite a downstream reader to count one hypothesis
    twice, so it is withheld rather than computed and ignored.
    """
    interval = _bootstrap_rate_delta(
        baseline_gt_status,
        candidate_gt_status,
        images,
        flag_column="detected",
        metric="recall",
        iterations=iterations,
        seed=seed,
        diagnostic=diagnostic,
    )
    return interval.model_copy(update={"p_value": float("nan")})


def validate_q(q: float) -> None:
    """Reject unsupported false discovery probabilities before computation."""
    if not math.isfinite(q) or not 0 <= q <= 1:
        raise ValueError("q must be a finite probability in [0, 1]")


def adjust_benjamini_hochberg(p_values: Sequence[float], q: float = 0.05) -> BHResult:
    """Control the false discovery rate across every class-by-metric test in a split.

    Undefined tests arrive as NaN and are held out of the family entirely - leaving them
    in would inflate its size and distort the ranking - then mapped back as NaN so the
    output stays aligned to the input order.
    """
    validate_q(q)
    raw = np.asarray(p_values, dtype=float)
    defined = np.isfinite(raw)
    adjusted = np.full(raw.shape, np.nan)
    if defined.any():
        adjusted[defined] = false_discovery_control(raw[defined], method="bh")
    return BHResult(
        adjusted_p_values=[float(value) for value in adjusted],
        rejected=[bool(value <= q) for value in adjusted],
        q=q,
    )


class PairedClassResult(BaseModel):
    """Precision test, recall interval and authoritative exact recall test."""

    precision: TestResult
    recall_interval: TestResult
    recall: TestResult


class PairedComparisonResult(BaseModel):
    """Caller-selected class results and pooled summary outside the BH family."""

    by_class: dict[str, PairedClassResult]
    pooled: PairedClassResult
    adjusted: BHResult


def compare_paired(
    baseline_gt_status: pd.DataFrame,
    baseline_pred_status: pd.DataFrame,
    candidate_gt_status: pd.DataFrame,
    candidate_pred_status: pd.DataFrame,
    images: Sequence[str],
    *,
    classes: Sequence[str] | None = None,
    iterations: int = 10_000,
    seed: int = 0,
    q: float = 0.05,
    diagnostic: Callable[[str], None] | None = None,
) -> PairedComparisonResult:
    """Compare supplied observations with exact McNemar and paired image bootstrap.

    ``classes=None`` computes one pooled pair over all supplied rows, no class
    tests and an empty BH family. An explicit sequence selects the eligible
    population; every class contributes precision and recall to one BH family
    (all precision p-values, then all recall p-values). The pooled result uses
    exactly that population and is outside the family. Eligibility and verdict
    policy belong to the caller. Empty explicit classes produce unavailable
    pooled tests, without attempting bootstrap over an empty population.
    """
    validate_q(q)
    by_class: dict[str, PairedClassResult] = {}
    if classes is not None:
        if len(set(classes)) != len(classes):
            raise ValueError("classes must not contain duplicates")
        for name in classes:
            by_class[name] = _paired_tests(
                baseline_gt_status[baseline_gt_status["instance_label"] == name],
                baseline_pred_status[baseline_pred_status["instance_label"] == name],
                candidate_gt_status[candidate_gt_status["instance_label"] == name],
                candidate_pred_status[candidate_pred_status["instance_label"] == name],
                images,
                iterations,
                seed,
                diagnostic,
            )
    adjusted = adjust_benjamini_hochberg(
        [result.precision.p_value for result in by_class.values()]
        + [result.recall.p_value for result in by_class.values()],
        q=q,
    )
    if classes is None:
        pooled = _paired_tests(
            baseline_gt_status,
            baseline_pred_status,
            candidate_gt_status,
            candidate_pred_status,
            images,
            iterations,
            seed,
            diagnostic,
        )
    elif classes:
        pooled = _paired_tests(
            baseline_gt_status[baseline_gt_status["instance_label"].isin(classes)],
            baseline_pred_status[baseline_pred_status["instance_label"].isin(classes)],
            candidate_gt_status[candidate_gt_status["instance_label"].isin(classes)],
            candidate_pred_status[candidate_pred_status["instance_label"].isin(classes)],
            images,
            iterations,
            seed,
            diagnostic,
        )
    else:
        unavailable = TestResult(delta=math.nan, p_value=math.nan, n_baseline=0, n_candidate=0)
        pooled = PairedClassResult(
            precision=unavailable, recall_interval=unavailable, recall=unavailable
        )
    return PairedComparisonResult(by_class=by_class, pooled=pooled, adjusted=adjusted)


def _paired_tests(
    baseline_gt: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    candidate_gt: pd.DataFrame,
    candidate_preds: pd.DataFrame,
    images: Sequence[str],
    iterations: int,
    seed: int,
    diagnostic: Callable[[str], None] | None,
) -> PairedClassResult:
    precision = bootstrap_precision_delta(
        baseline_preds,
        candidate_preds,
        images,
        iterations=iterations,
        seed=seed,
        diagnostic=diagnostic,
    )
    recall_interval = bootstrap_recall_delta(
        baseline_gt,
        candidate_gt,
        images,
        iterations=iterations,
        seed=seed,
        diagnostic=diagnostic,
    )
    recall = mcnemar_recall(
        baseline_gt.set_index("gt_index")["detected"],
        candidate_gt.set_index("gt_index")["detected"],
        diagnostic=diagnostic,
    )
    return PairedClassResult(precision=precision, recall_interval=recall_interval, recall=recall)
