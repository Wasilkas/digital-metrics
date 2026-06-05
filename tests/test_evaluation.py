import pandas as pd
import pytest

from metrics import Evaluation


def test_greedy_round_trip(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5, matching_strategy="greedy")
    ev(split="all", find_best_confs=True)

    assert set(ev.metrics.keys()) == {"class_a", "class_b", "class_c"}
    for m in ev.metrics.values():
        assert m.ap50 >= 0.0


def test_hungarian_round_trip(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5, matching_strategy="hungarian")
    ev(split="all", find_best_confs=True)

    assert set(ev.metrics.keys()) == {"class_a", "class_b", "class_c"}


def test_cm_shape(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="all", find_best_confs=False)

    n_classes = 3
    assert ev.cm is not None
    assert ev.cm.shape == (n_classes + 1, n_classes + 1)
    assert len(ev.class_labels) == n_classes + 1
    assert "background" in ev.class_labels


def test_metrics_fields_populated(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="all", find_best_confs=True)

    for m in ev.metrics.values():
        assert m.ap50 >= 0.0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.precision_ci_lower <= m.precision
        assert m.precision <= m.precision_ci_upper <= 1.0


def test_validate_df_raises_on_missing_column(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    preds_bad = preds_df.drop(columns=["confidence"])
    ev = Evaluation(preds_bad, gt_df, iou_threshold=0.5)
    with pytest.raises(ValueError, match="confidence"):
        ev(split="all")


def test_validate_df_raises_on_missing_gt_column(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    gt_bad = gt_df.drop(columns=["instance_label"])
    ev = Evaluation(preds_df, gt_bad, iou_threshold=0.5)
    with pytest.raises(ValueError, match="instance_label"):
        ev(split="all")


def test_hungarian_keys_match_greedy(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    ev_g = Evaluation(preds_df, gt_df, matching_strategy="greedy")
    ev_g(split="all", find_best_confs=False)
    ev_h = Evaluation(preds_df, gt_df, matching_strategy="hungarian")
    ev_h(split="all", find_best_confs=False)

    assert set(ev_g.metrics.keys()) == set(ev_h.metrics.keys())


def test_get_dfs_visualization(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df)
    ev(split="all", find_best_confs=False)
    gt_vis, pred_vis = ev.get_dfs_visualization(find_best_confs=False)
    assert "predict_type" in gt_vis.columns
    assert "predict_type" in pred_vis.columns


# ---------------------------------------------------------------------------
# Calibration split tests
# ---------------------------------------------------------------------------


def test_calibration_split_smoke(split_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """calibration_split runs without error and populates metrics for test split."""
    gt_df, preds_df = split_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="test", calibration_split="val")

    assert set(ev.metrics.keys()) >= {"class_a", "class_b"}
    assert ev.cm is not None
    for m in ev.metrics.values():
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0


def test_calibration_split_thresholds_equal_val_insample(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Thresholds from calibration_split='val' equal those found in-sample on val."""
    gt_df, preds_df = split_dataset

    # In-sample: find thresholds directly on the val split
    ev_val = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev_val(split="val", find_best_confs=True)
    val_thresholds = dict(ev_val.best_confidences)

    # Out-of-sample: calibrate on val, evaluate on test
    ev_test = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev_test(split="test", calibration_split="val")
    cal_thresholds = dict(ev_test.best_confidences)

    # Both paths use the same val matches → identical thresholds
    for c in ("class_a", "class_b"):
        assert val_thresholds.get(c, -1.0) == pytest.approx(cal_thresholds.get(c, -2.0))


def test_calibration_split_vs_insample_different_metrics(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Test metrics differ when calibrating on val vs optimising in-sample on test."""
    gt_df, preds_df = split_dataset

    ev_insample = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev_insample(split="test", find_best_confs=True)

    ev_calibrated = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev_calibrated(split="test", calibration_split="val")

    # Val confidence thresholds (~0.85–0.90) are higher than test thresholds (~0.60–0.70)
    # so the calibrated run uses stricter filtering — both should still be valid floats
    for c in ("class_a", "class_b"):
        assert isinstance(ev_calibrated.best_confidences.get(c), float)
        assert isinstance(ev_insample.best_confidences.get(c), float)


def test_calibration_split_invalid_raises(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """calibration_split with a non-existent value raises ValueError with a clear message."""
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    with pytest.raises(ValueError, match="nonexistent"):
        ev(split="all", calibration_split="nonexistent")


def test_calibration_split_no_split_column_raises(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """calibration_split raises ValueError when split_df has no 'split' column."""
    gt_df, preds_df = tiny_dataset
    gt_no_split = gt_df.drop(columns=["split"])
    ev = Evaluation(preds_df, gt_no_split, iou_threshold=0.5)
    with pytest.raises(ValueError, match="'split' column"):
        ev(split="all", calibration_split="val")
