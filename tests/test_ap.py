import numpy as np
import pandas as pd
import pytest

from metrics.ap import APMethod, compute_ap, compute_map
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


def test_compute_map_empty_images_count_as_fp() -> None:
    """Predictions on empty split images (no GT at all) must be counted as FP.

    Without split_image_names, empty images are invisible (they have no rows
    in gt_df) so their predictions are silently dropped, inflating AP.
    With split_image_names the detections on empty images must be counted as
    FPs and AP must fall below 1.0.

    The FP must be the *highest*-confidence prediction so it appears first in
    the confidence-sorted curve and genuinely lowers AP.  A low-confidence FP
    ranked after the TP cannot lower AP because recall is already at its max.
    """
    cols_gt = [
        "image_name", "instance_label",
        "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "split",
    ]
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "val")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name", "instance_label",
        "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "confidence",
    ]
    preds = pd.DataFrame(
        [
            # FP first in confidence ranking so it sits before the TP on the P-R curve.
            ("img_empty", "cat", 10, 10, 110, 110, 0.99),  # FP — empty image, high conf
            ("img1", "cat", 0, 0, 100, 100, 0.50),         # TP — annotated image, lower conf
        ],
        columns=cols_p,
    )

    # Without split_image_names: img_empty is invisible, only TP is scored → AP = 1.0 (wrong).
    metrics_no_empty: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics_no_empty)
    assert metrics_no_empty["cat"].ap50 == pytest.approx(1.0, abs=1e-6)

    # With split_image_names: the high-confidence FP on img_empty is counted → AP = 0.5.
    metrics_with_empty: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics_with_empty, split_image_names=["img1", "img_empty"])
    assert metrics_with_empty["cat"].ap50 == pytest.approx(0.5, abs=1e-6)


# --- interp (101-point COCO) method tests ---


@pytest.mark.parametrize("method", ["continuous", "interp"])
def test_ap_perfect_detector_both_methods(method: APMethod) -> None:
    recall = np.array([0.1, 0.2, 0.5, 1.0])
    precision = np.array([1.0, 1.0, 1.0, 1.0])
    assert compute_ap(recall, precision, method) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("method", ["continuous", "interp"])
def test_ap_zero_precision_both_methods(method: APMethod) -> None:
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([0.0, 0.0, 0.0])
    assert compute_ap(recall, precision, method) == pytest.approx(0.0, abs=1e-6)


def test_ap_interp_empty_recall_returns_zero() -> None:
    # interp method must not crash on empty arrays (no predictions made)
    assert compute_ap(np.array([]), np.array([]), method="interp") == pytest.approx(0.0)


def test_ap_interp_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    recall = np.sort(rng.uniform(0, 1, 20))
    precision = rng.uniform(0, 1, 20)
    ap = compute_ap(recall, precision, method="interp")
    assert 0.0 <= ap <= 1.0


def test_compute_map_interp_perfect() -> None:
    gt, preds = _make_perfect_dataset()
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics, method="interp")
    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-3)
    assert metrics["cat"].ap75 == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= metrics["cat"].ap50_95 <= 1.0


def test_compute_map_interp_all_fp() -> None:
    cols_gt = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "split"]
    gt = pd.DataFrame([("img1", "cat", 0, 0, 100, 100, "test")], columns=cols_gt)
    cols_p = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "confidence"]
    preds = pd.DataFrame([("img1", "cat", 500, 500, 600, 600, 0.9)], columns=cols_p)
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics, method="interp")
    assert metrics["cat"].ap50 == pytest.approx(0.0, abs=1e-3)


def test_compute_map_strategy_greedy_vs_iou_prior() -> None:
    """Greedy and iou_prior give different AP when confidence and IoU rankings disagree.

    One GT; pred_a has higher confidence but lower IoU than pred_b.
    Greedy: pred_a wins the GT (TP first) → AP=1.0 at IoU=0.5.
    IoU-prior: pred_b wins the GT (higher IoU) → pred_a is FP first in the
    confidence-sorted curve → AP=0.5.
    """
    cols_gt = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "split"]
    gt = pd.DataFrame([("img", "cat", 0, 0, 100, 100, "test")], columns=cols_gt)
    cols_p = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "confidence"]
    preds = pd.DataFrame(
        [
            ("img", "cat", 0, 0, 80, 100, 0.9),   # IoU=0.80, high conf
            ("img", "cat", 0, 0, 100, 100, 0.5),  # IoU=1.00, low conf
        ],
        columns=cols_p,
    )

    m_greedy: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, m_greedy, strategy="greedy")
    assert m_greedy["cat"].ap50 == pytest.approx(1.0, abs=1e-6)

    m_iou: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, m_iou, strategy="iou_prior")
    assert m_iou["cat"].ap50 == pytest.approx(0.5, abs=1e-3)


def test_compute_map_strategy_hungarian_perfect() -> None:
    """Hungarian matching on a perfect detector yields AP=1.0."""
    gt, preds = _make_perfect_dataset()
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics, strategy="hungarian")
    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-6)
    assert metrics["cat"].ap75 == pytest.approx(1.0, abs=1e-6)


def test_compute_map_strategy_hungarian_iou_optimal() -> None:
    """Hungarian picks the higher-IoU pred regardless of confidence (like iou_prior).

    One GT; pred_a has higher confidence but lower IoU than pred_b.
      greedy:    pred_a wins GT (confidence-sorted first) → TP at high conf → AP=1.0
      iou_prior: pred_b wins GT (higher IoU)              → FP at high conf → AP=0.5
      hungarian: pred_b wins GT (globally optimal)        → FP at high conf → AP=0.5
    """
    cols_gt = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "split"]
    gt = pd.DataFrame([("img", "cat", 0, 0, 100, 100, "test")], columns=cols_gt)
    cols_p = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "confidence"]
    preds = pd.DataFrame(
        [
            ("img", "cat", 0, 0, 80, 100, 0.9),   # pred_a: IoU=0.80, high conf
            ("img", "cat", 0, 0, 100, 100, 0.5),  # pred_b: IoU=1.00, low conf
        ],
        columns=cols_p,
    )

    m_hungarian: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, m_hungarian, strategy="hungarian")
    assert m_hungarian["cat"].ap50 == pytest.approx(0.5, abs=1e-3)


def test_ap_interp_lower_than_continuous_for_imperfect_curve() -> None:
    # For a non-trivial PR curve the two methods produce different values.
    # Neither is universally larger; we just verify they differ.
    recall = np.array([0.2, 0.4, 0.6, 0.8])
    precision = np.array([0.9, 0.7, 0.5, 0.3])
    ap_cont = compute_ap(recall, precision, method="continuous")
    ap_interp = compute_ap(recall, precision, method="interp")
    assert ap_cont != pytest.approx(ap_interp, abs=1e-4)
