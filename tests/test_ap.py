import numpy as np
import pandas as pd
import pytest

from metrics.ap import compute_ap, compute_map
from metrics.types import Metrics


def test_ap_perfect_detector() -> None:
    # Recall rises from 0→1 with precision held at 1 → AP = 1.0
    recall = np.array([0.1, 0.2, 0.5, 1.0])
    precision = np.array([1.0, 1.0, 1.0, 1.0])
    assert compute_ap(recall, precision) == pytest.approx(1.0)


def test_ap_zero_precision() -> None:
    # All predictions are wrong; precision = 0 everywhere → AP = 0
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([0.0, 0.0, 0.0])
    assert compute_ap(recall, precision) == pytest.approx(0.0)


def test_ap_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    recall = np.sort(rng.uniform(0, 1, 20))
    precision = rng.uniform(0, 1, 20)
    ap = compute_ap(recall, precision)
    assert 0.0 <= ap <= 1.0


def _make_perfect_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfect detector: one class, preds exactly match GTs."""
    cols_gt = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "split",
    ]
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "test"), ("img2", "cat", 0, 0, 100, 100, "test")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
    ]
    preds = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, 0.99), ("img2", "cat", 0, 0, 100, 100, 0.95)],
        columns=cols_p,
    )
    return gt, preds


def test_compute_map_perfect() -> None:
    gt, preds = _make_perfect_dataset()
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics)
    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-6)
    assert metrics["cat"].ap75 == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= metrics["cat"].ap50_95 <= 1.0


def test_compute_map_all_fp() -> None:
    """All predictions are FP (wrong location) → mAP ≈ 0."""
    cols_gt = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "split",
    ]
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "test")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
    ]
    preds = pd.DataFrame(
        [("img1", "cat", 500, 500, 600, 600, 0.9)],  # no overlap with GT
        columns=cols_p,
    )
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics)
    assert metrics["cat"].ap50 == pytest.approx(0.0, abs=1e-6)


def test_compute_map_ignores_predictions_outside_split() -> None:
    """Predictions for images outside gt_df must not be scored as FP.

    Real workflows often run inference once over the whole dataset, so
    preds_df may contain rows for train/test images while gt_df is scoped
    to e.g. "val". Those foreign predictions must be dropped before scoring,
    not counted as false positives (which would crater AP) nor interleaved
    into the confidence-sorted precision-recall curve.
    """
    cols_gt = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "split",
    ]
    # Only "img1" belongs to the evaluated split; "img_other" does not appear
    # in gt_df at all (e.g. it's a train/test image).
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "val")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
    ]
    preds = pd.DataFrame(
        [
            ("img1", "cat", 0, 0, 100, 100, 0.99),  # perfect match → TP
            # Foreign predictions for an image outside this split, at very
            # high confidence so they would dominate a confidence-sorted
            # cumulative curve and would be marked FP if not filtered out.
            ("img_other", "cat", 10, 10, 110, 110, 0.999),
            ("img_other", "cat", 200, 200, 300, 300, 0.998),
        ],
        columns=cols_p,
    )
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics)

    # A single TP and nothing else in-split → perfect AP, unaffected by the
    # foreign high-confidence predictions for img_other.
    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-6)
    assert metrics["cat"].ap75 == pytest.approx(1.0, abs=1e-6)


def test_compute_map_class_absent_from_split_is_nan() -> None:
    """A class with zero GT instances in this split must be NaN, not 0.0.

    Evaluation.metrics is keyed by all classes in split_df (across every
    split), but compute_map only scores classes present in the *current*
    split's GT. Classes absent here were never evaluated for AP — reporting
    0.0 would silently drag down a class-averaged mAP (nanmean must skip them
    to match how YOLO reports mAP50: mean only over classes seen in val).
    """
    gt, preds = _make_perfect_dataset()
    # "dog" never appears in gt — simulates a class present in split_df overall
    # (e.g. only in train/test) but absent from this evaluation split.
    metrics: dict[str, Metrics] = {"cat": Metrics(), "dog": Metrics()}
    compute_map(gt, preds, metrics)

    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-6)
    assert np.isnan(metrics["dog"].ap50)
    assert np.isnan(metrics["dog"].ap75)
    assert np.isnan(metrics["dog"].ap50_95)

    # nanmean over all classes must equal the score of the evaluated class only
    assert np.nanmean([metrics["cat"].ap50, metrics["dog"].ap50]) == pytest.approx(1.0)
