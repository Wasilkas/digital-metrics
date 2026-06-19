import random

import pandas as pd
import pytest

from metrics import Evaluation
from metrics.matching import match_boxes
from metrics.scoring import (
    find_best_confidences,
    find_best_global_confidence,
    slice_by_conf,
)
from metrics.types import PredictMatch


def _mk(pred_label: str, gt_label: str, conf: float) -> PredictMatch:
    return PredictMatch(
        pred_label=pred_label, gt_label=gt_label, pred_index=0, gt_index=0, confidence=conf
    )


def _realized_f1(matches: dict[str, list[PredictMatch]], c: str, thr: float) -> float:
    """F1 actually obtained when the threshold is applied via slice_by_conf."""
    sliced = slice_by_conf({c: matches[c]}, [c], {c: thr})[c]
    tp = sum(1 for m in sliced if m.type == "TP")
    fp = sum(1 for m in sliced if m.type == "FP")
    fn = sum(1 for m in sliced if m.type == "FN")
    p = tp / max(tp + fp, 1e-9)
    r = tp / max(tp + fn, 1e-9)
    return 2 * p * r / max(p + r, 1e-9)


def _tiny_matches(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    gt_df, preds_df = tiny_dataset
    classes = gt_df["instance_label"].unique().tolist()
    matches = match_boxes(
        gt_df,
        preds_df,
        iou_threshold=0.5,
        strategy="greedy",
        split_image_names=gt_df["image_name"].unique().tolist(),
    )
    return matches, classes


def test_per_class_thresholds_differ_across_classes(tiny_dataset) -> None:
    matches, classes = _tiny_matches(tiny_dataset)

    per_class = find_best_confidences(matches, classes)

    # class_a peaks at conf=0.85 (all 3 TP, no FP), class_c only at 0.60.
    assert per_class["class_a"] == 0.85
    assert per_class["class_c"] == 0.60
    assert len(set(per_class.values())) > 1  # genuinely per-class


def test_global_confidence_is_single_shared_threshold(tiny_dataset) -> None:
    matches, classes = _tiny_matches(tiny_dataset)

    threshold = find_best_global_confidence(matches, classes)

    # Mean per-class F1 is maximised at 0.60: class_a is perfect (P=R=1),
    # class_b and class_c still keep their single TP.
    assert threshold == 0.60


def test_global_confidence_empty_matches_returns_zero() -> None:
    assert find_best_global_confidence({}, ["class_a"]) == 0.0


def test_evaluation_global_mode_applies_one_threshold(tiny_dataset) -> None:
    gt_df, preds_df = tiny_dataset

    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5, confidence_optimization="global")
    ev(split="test", find_best_confs=True)

    confidences = set(ev.best_confidences.values())
    assert len(confidences) == 1  # YOLO-style: one threshold for every class
    assert confidences == {0.60}


def test_evaluation_per_class_mode_is_default(tiny_dataset) -> None:
    gt_df, preds_df = tiny_dataset

    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="test", find_best_confs=True)

    assert len(set(ev.best_confidences.values())) > 1


# ---------------------------------------------------------------------------
# Confidence-optimization correctness (realizable thresholds, tie handling)
# ---------------------------------------------------------------------------


def test_find_best_confidences_handles_tied_confidences() -> None:
    """Tied confidences must not yield a non-realizable operating point.

    3 TP at 0.9, then 1 TP + 2 FP all tied at 0.8.  Thresholding keeps every
    detection with confidence >= t, so the only achievable operating points are
    "keep >= 0.9" (P=1.0, R=0.75, F1=0.857) and "keep >= 0.8" (P=4/6, R=1.0,
    F1=0.80).  A per-detection cumulative sweep could split the 0.8 group and
    report an inflated F1 at 0.8 — the fix must return the realizable optimum
    0.9.
    """
    matches = {
        "c": [
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.8),
            _mk("c", "background", 0.8),
            _mk("c", "background", 0.8),
        ]
    }

    best = find_best_confidences(matches, ["c"])["c"]

    assert best == 0.9
    # The chosen threshold is the realizable argmax, not the inflated 0.8 point.
    assert _realized_f1(matches, "c", 0.9) > _realized_f1(matches, "c", 0.8)


def test_find_best_confidences_picks_global_realizable_optimum() -> None:
    """Differential check: the chosen threshold's *realized* F1 equals the best
    realized F1 over all candidate thresholds, across many random match sets."""
    rng = random.Random(0)

    for _ in range(300):
        ms: list[PredictMatch] = []
        for _ in range(rng.randint(1, 6)):
            x = rng.random()
            conf = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9])
            if x < 0.4:
                ms.append(_mk("c", "c", conf))  # TP
            elif x < 0.7:
                ms.append(_mk("c", "background", conf))  # FP
            else:
                ms.append(_mk("background", "c", 0.0))  # FN
        rng.shuffle(ms)
        matches = {"c": ms}

        got = find_best_confidences(matches, ["c"])["c"]

        thresholds = {m.confidence for m in ms if m.type != "FN"}
        if not thresholds:
            assert got == 0.0
            continue

        best_realized = max(_realized_f1(matches, "c", t) for t in thresholds)
        assert _realized_f1(matches, "c", got) == pytest.approx(best_realized)


def test_find_best_confidences_no_detections_returns_zero() -> None:
    # Only FN entries (no predictions for the class) → threshold defaults to 0.0.
    matches = {"c": [_mk("background", "c", 0.0), _mk("background", "c", 0.0)]}
    assert find_best_confidences(matches, ["c"])["c"] == 0.0
    assert find_best_confidences({}, ["c"])["c"] == 0.0


def test_global_confidence_handles_tied_confidences() -> None:
    """The global sweep evaluates each distinct threshold with '>= t' semantics,
    so tied confidences are grouped correctly (no mid-tie operating point)."""
    matches = {
        "c": [
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.9),
            _mk("c", "c", 0.8),
            _mk("c", "background", 0.8),
            _mk("c", "background", 0.8),
        ]
    }
    assert find_best_global_confidence(matches, ["c"]) == 0.9
