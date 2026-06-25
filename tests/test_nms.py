import pandas as pd
import pytest

from digital_metrics.evaluation import Evaluation
from digital_metrics.preprocess import apply_nms, filter_by_confidence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GT_COLS = [
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
    "split",
]
_PRED_COLS = [
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
    "confidence",
]


def _preds(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=_PRED_COLS)


def _gt(*rows: tuple) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=_GT_COLS)


# ---------------------------------------------------------------------------
# filter_by_confidence
# ---------------------------------------------------------------------------


def test_filter_by_confidence_removes_below_threshold() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 10, 10, 0.9),
        ("img1", "cat", 0, 0, 10, 10, 0.5),
        ("img1", "cat", 0, 0, 10, 10, 0.3),
    )
    result = filter_by_confidence(df, threshold=0.5)
    assert len(result) == 2
    assert set(result["confidence"].tolist()) == {0.9, 0.5}


def test_filter_by_confidence_keeps_all_when_threshold_zero() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 10, 10, 0.1),
        ("img1", "cat", 0, 0, 10, 10, 0.0),
    )
    result = filter_by_confidence(df, threshold=0.0)
    assert len(result) == 2


def test_filter_by_confidence_removes_all_when_threshold_above_max() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 10, 10, 0.8),
        ("img1", "cat", 0, 0, 10, 10, 0.6),
    )
    result = filter_by_confidence(df, threshold=0.9)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# apply_nms — same-class containment
# ---------------------------------------------------------------------------


def test_nms_same_class_small_inside_large_is_suppressed() -> None:
    """Small box fully inside a large same-class box → small one suppressed."""
    df = _preds(
        # large box, higher confidence → kept
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        # small box fully inside the large one → suppressed
        ("img1", "cat", 10, 10, 50, 50, 0.7),
    )
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=1.01)
    assert len(result) == 1
    assert result.iloc[0]["confidence"] == pytest.approx(0.9)


def test_nms_same_class_no_overlap_both_kept() -> None:
    """Two same-class boxes with no overlap → both kept."""
    df = _preds(
        ("img1", "cat", 0, 0, 50, 50, 0.9),
        ("img1", "cat", 200, 200, 250, 250, 0.7),
    )
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=1.01)
    assert len(result) == 2


def test_nms_same_class_partial_overlap_below_threshold_both_kept() -> None:
    """Partial overlap below containment threshold → both boxes kept."""
    df = _preds(
        # overlapping but neither is mostly inside the other
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "cat", 60, 60, 160, 160, 0.7),
    )
    # containment = intersection / min_area  = 40*40 / (100*100) = 0.16
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=1.01)
    assert len(result) == 2


def test_nms_same_class_higher_confidence_wins() -> None:
    """When the smaller (inner) box has higher confidence it still wins because
    the larger box (lower confidence) comes second and is suppressed."""
    df = _preds(
        # small inner box — higher confidence, processed first
        ("img1", "cat", 10, 10, 50, 50, 0.95),
        # large outer box — lower confidence
        ("img1", "cat", 0, 0, 100, 100, 0.60),
    )
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=1.01)
    assert len(result) == 1
    assert result.iloc[0]["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# apply_nms — cross-class IoU
# ---------------------------------------------------------------------------


def test_nms_cross_class_high_iou_lower_confidence_suppressed() -> None:
    """Two different-class boxes with IoU > threshold → lower confidence dropped."""
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "dog", 5, 5, 105, 105, 0.7),  # high overlap, lower conf
    )
    result = apply_nms(df, same_class_containment_threshold=1.01, cross_class_iou_threshold=0.5)
    assert len(result) == 1
    assert result.iloc[0]["instance_label"] == "cat"


def test_nms_cross_class_low_iou_both_kept() -> None:
    """Two different-class boxes with IoU below threshold → both kept."""
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "dog", 80, 80, 180, 180, 0.7),  # small overlap
    )
    # IoU = 20*20 / (100*100 + 100*100 - 20*20) = 400/19600 ≈ 0.02
    result = apply_nms(df, same_class_containment_threshold=1.01, cross_class_iou_threshold=0.5)
    assert len(result) == 2


def test_nms_cross_class_no_suppression_when_threshold_above_one() -> None:
    """Cross-class threshold > 1 disables cross-class NMS entirely."""
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "dog", 0, 0, 100, 100, 0.8),  # perfect overlap
    )
    result = apply_nms(df, same_class_containment_threshold=1.01, cross_class_iou_threshold=1.01)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# apply_nms — multi-image isolation
# ---------------------------------------------------------------------------


def test_nms_suppression_is_per_image() -> None:
    """A box that would be suppressed on img1 is kept on img2 independently."""
    df = _preds(
        # img1: two same-class boxes — small inside large
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "cat", 10, 10, 50, 50, 0.7),
        # img2: same geometry but no second box — both should survive independently
        ("img2", "cat", 0, 0, 100, 100, 0.9),
    )
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=1.01)
    img1_result = result[result["image_name"] == "img1"]
    img2_result = result[result["image_name"] == "img2"]
    assert len(img1_result) == 1  # inner box suppressed
    assert len(img2_result) == 1  # nothing to suppress


def test_nms_empty_image_no_predictions_ok() -> None:
    """An image with no predictions doesn't cause errors."""
    df = _preds(("img1", "cat", 0, 0, 100, 100, 0.9))
    # preds_df has only img1; apply_nms over it should work fine
    result = apply_nms(df, same_class_containment_threshold=0.8, cross_class_iou_threshold=0.5)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------


def _make_eval_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = _gt(
        ("img1", "cat", 0, 0, 100, 100, "test"),
        ("img1", "dog", 200, 200, 300, 300, "test"),
    )
    preds = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),  # TP
        ("img1", "cat", 5, 5, 90, 90, 0.7),  # inside first cat box → should be NMS'd
        ("img1", "dog", 200, 200, 300, 300, 0.8),  # TP
        ("img1", "cat", 0, 0, 100, 100, 0.1),  # below conf threshold
    )
    return gt, preds


def test_evaluation_preprocess_preds_conf_threshold_removes_rows() -> None:
    gt, preds = _make_eval_data()
    ev = Evaluation(preds, gt, preprocess_preds_conf_threshold=0.5)
    # Row with conf=0.1 should be removed
    assert len(ev.preds_df) == 3
    assert all(ev.preds_df["confidence"] >= 0.5)


def test_evaluation_preprocess_preds_nms_removes_contained_box() -> None:
    gt, preds = _make_eval_data()
    ev = Evaluation(preds, gt, preprocess_preds_nms_containment_threshold=0.8)
    # The inner cat box (conf=0.7) should be suppressed by the outer cat box (conf=0.9)
    cat_preds = ev.preds_df[ev.preds_df["instance_label"] == "cat"]
    assert len(cat_preds) == 1
    assert cat_preds.iloc[0]["confidence"] == pytest.approx(0.9)


def test_evaluation_preprocess_preds_conf_and_nms_combined() -> None:
    gt, preds = _make_eval_data()
    ev = Evaluation(
        preds,
        gt,
        preprocess_preds_conf_threshold=0.5,
        preprocess_preds_nms_containment_threshold=0.8,
    )
    # conf filter removes conf=0.1; NMS removes inner cat box (conf=0.7)
    assert len(ev.preds_df) == 2
    assert set(ev.preds_df["instance_label"].tolist()) == {"cat", "dog"}


def test_evaluation_no_preprocess_preds_unchanged() -> None:
    gt, preds = _make_eval_data()
    ev = Evaluation(preds, gt)
    assert len(ev.preds_df) == len(preds)
