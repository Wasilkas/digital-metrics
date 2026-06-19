"""Profile all three metric paths on the same fixture boxes.

Three "ways" to score identical GT / prediction boxes on the test split:

1. ``Evaluation`` — this library's own pipeline at the documented YOLO recipe
   (``matching_strategy="iou_prior"``, ``ap_method="interp"``,
   ``confidence_optimization="global"``, in-sample threshold on test).
2. ``ultralytics`` — ``compute_detection_metrics(backend="ultralytics")``.
3. ``torchmetrics`` — ``compute_detection_metrics(backend="torchmetrics")``.

For each way it reports:
* wall-clock: one warmup run (load torch / JIT caches), then ``--runs`` timed
  runs; mean / min / stdev.
* cProfile: one run under ``cProfile``, top functions by cumulative time.

External backends are optional extras (both pull in torch); a backend whose
dependency is missing is skipped with a note.

Run (with both extras):
    uv run python scripts/profile_backends.py
    uv run python scripts/profile_backends.py --runs 5 --top 20
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from metrics import Backend, Evaluation, compute_detection_metrics

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"

SPLIT = "test"
IOU_THRESHOLD = 0.5


def run_ours(preds_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    """One full native Evaluation scoring pass (YOLO recipe)."""
    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=IOU_THRESHOLD,
        matching_strategy="iou_prior",
        ap_method="interp",
        confidence_optimization="global",
    )
    ev(split=SPLIT, find_best_confs=True)


def run_external(
    backend: Backend, test_gt: pd.DataFrame, test_preds: pd.DataFrame, classes: list[str]
) -> None:
    """One full external-backend scoring pass."""
    compute_detection_metrics(test_gt, test_preds, backend=backend, classes=classes)


def time_way(fn: Callable[[], None], runs: int) -> tuple[float, float, float] | None:
    """Warmup once, then time ``runs`` calls. Returns (mean, min, stdev) seconds."""
    try:
        fn()  # warmup: torch import, JIT, lazy caches
    except ImportError as exc:
        logger.warning(f"skipped — {exc}")
        return None
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.mean(samples), min(samples), stdev


def profile_way(fn: Callable[[], None], top: int) -> str:
    """Run ``fn`` once under cProfile; return top-``top`` functions by cumtime."""
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(top)
    return buf.getvalue()


def print_timing(results: dict[str, tuple[float, float, float] | None]) -> None:
    print("\n" + "=" * 58)
    print(f"{'WAY':<14}{'mean (s)':>12}{'min (s)':>12}{'stdev':>12}{'rel':>8}")
    print("=" * 58)
    fastest = min((r[1] for r in results.values() if r), default=0.0)
    for name, r in results.items():
        if r is None:
            print(f"{name:<14}{'skipped (extra missing)':>44}")
            continue
        mean, mn, sd = r
        rel = mn / fastest if fastest else 1.0
        print(f"{name:<14}{mean:>12.4f}{mn:>12.4f}{sd:>12.4f}{rel:>7.1f}x")
    print("=" * 58)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=int, default=3, help="timed runs per way (after warmup)")
    ap.add_argument("--top", type=int, default=15, help="top N functions in cProfile output")
    ap.add_argument("--no-profile", action="store_true", help="wall-clock only, skip cProfile")
    args = ap.parse_args()

    logger.info("Loading data...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)

    test_gt = gt_df[gt_df["split"] == SPLIT]
    test_images = set(test_gt["image_name"].unique())
    test_preds = preds_df[preds_df["image_name"].isin(test_images)]
    classes = sorted(gt_df["instance_label"].unique())
    logger.info(
        f"test: {test_gt['image_name'].nunique()} images, "
        f"{len(test_gt)} GT boxes, {len(test_preds)} predictions; runs={args.runs}"
    )

    ways: dict[str, Callable[[], None]] = {
        "ours": lambda: run_ours(preds_df, gt_df),
        "ultralytics": lambda: run_external("ultralytics", test_gt, test_preds, classes),
        "torchmetrics": lambda: run_external("torchmetrics", test_gt, test_preds, classes),
    }

    logger.info("Wall-clock timing (warmup + timed runs)...")
    results: dict[str, tuple[float, float, float] | None] = {}
    for name, fn in ways.items():
        logger.info(f"[{name}] timing...")
        results[name] = time_way(fn, args.runs)
    print_timing(results)

    if args.no_profile:
        return
    for name, fn in ways.items():
        if results[name] is None:
            continue  # extra missing
        logger.info(f"[{name}] cProfile...")
        print(f"\n{'#' * 58}\n# cProfile — {name} (top {args.top} by cumulative time)\n{'#' * 58}")
        print(profile_way(fn, args.top))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
