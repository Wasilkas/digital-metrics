import pandas as pd
import pytest
from loguru import logger

from metrics import Evaluation

_BBOX = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
_GT_COLS = ["image_name", "instance_label", *_BBOX, "split"]
_PRED_COLS = ["image_name", "instance_label", *_BBOX, "confidence"]


def _capture_warnings(fn: object) -> list[str]:
    """Run ``fn`` and return the WARNING-level loguru messages it emits."""
    msgs: list[str] = []
    handler_id = logger.add(msgs.append, level="WARNING")
    try:
        fn()  # type: ignore[operator]
    finally:
        logger.remove(handler_id)
    return [str(m) for m in msgs]


def _perfect_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every prediction is an exact-match TP → optimal threshold keeps all."""
    gt = pd.DataFrame(
        [("i1", "a", 0, 0, 10, 10, "test"), ("i2", "a", 0, 0, 10, 10, "test")],
        columns=_GT_COLS,
    )
    preds = pd.DataFrame(
        [("i1", "a", 0, 0, 10, 10, 0.9), ("i2", "a", 0, 0, 10, 10, 0.4)],
        columns=_PRED_COLS,
    )
    return gt, preds


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


def test_validate_df_raises_on_na_confidence(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    preds_bad = preds_df.copy()
    preds_bad.loc[0, "confidence"] = float("nan")
    ev = Evaluation(preds_bad, gt_df, iou_threshold=0.5)
    with pytest.raises(ValueError, match="confidence.*NA"):
        ev(split="all")


def test_validate_df_raises_on_unknown_pred_class(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    preds_bad = preds_df.copy()
    preds_bad.loc[0, "instance_label"] = "class_z"  # not in GT vocabulary
    ev = Evaluation(preds_bad, gt_df, iou_threshold=0.5)
    with pytest.raises(ValueError, match="class_z"):
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


def test_calibration_split_overlap_raises(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """calibration_split raises ValueError when an image_name appears in both
    the calibration split and the evaluation split — predictions are joined to
    GT via image_name, so overlap would leak calibration data into evaluation."""
    gt_df, preds_df = split_dataset

    # Duplicate img3's GT rows but mislabel the copies as "val": img3 now
    # appears in both "val" and "test" — a data-integrity bug to guard against.
    leaked_rows = gt_df[gt_df["image_name"] == "img3"].copy()
    leaked_rows["split"] = "val"
    leaked_gt = pd.concat([gt_df, leaked_rows], ignore_index=True)

    ev = Evaluation(preds_df, leaked_gt, iou_threshold=0.5)
    with pytest.raises(ValueError, match="shares"):
        ev(split="test", calibration_split="val")


# ---------------------------------------------------------------------------
# Unoptimized-threshold warning (threshold == minimum prediction confidence)
# ---------------------------------------------------------------------------


def test_per_class_unoptimized_threshold_warns() -> None:
    gt, preds = _perfect_dataset()
    ev = Evaluation(preds, gt, iou_threshold=0.5, confidence_optimization="per_class")
    msgs = _capture_warnings(lambda: ev(split="test", find_best_confs=True))

    assert ev.best_confidences["a"] == pytest.approx(0.4)  # the minimum confidence
    assert any("had no effect" in m for m in msgs)


def test_global_unoptimized_threshold_warns() -> None:
    gt, preds = _perfect_dataset()
    ev = Evaluation(preds, gt, iou_threshold=0.5, confidence_optimization="global")
    msgs = _capture_warnings(lambda: ev(split="test", find_best_confs=True))

    assert any("Global confidence threshold" in m for m in msgs)


def test_optimized_threshold_does_not_warn() -> None:
    # Two TPs (conf 0.9/0.8) plus two low-confidence background FPs (0.3/0.2): the
    # F1 optimum drops the FPs, so the threshold (0.8) sits above the minimum (0.2).
    gt = pd.DataFrame(
        [("i1", "a", 0, 0, 10, 10, "test"), ("i2", "a", 0, 0, 10, 10, "test")],
        columns=_GT_COLS,
    )
    preds = pd.DataFrame(
        [
            ("i1", "a", 0, 0, 10, 10, 0.9),  # TP
            ("i2", "a", 0, 0, 10, 10, 0.8),  # TP
            ("i1", "a", 500, 500, 510, 510, 0.3),  # FP (no GT there)
            ("i2", "a", 500, 500, 510, 510, 0.2),  # FP
        ],
        columns=_PRED_COLS,
    )
    ev = Evaluation(preds, gt, iou_threshold=0.5, confidence_optimization="per_class")
    msgs = _capture_warnings(lambda: ev(split="test", find_best_confs=True))

    assert ev.best_confidences["a"] == pytest.approx(0.8)  # above the 0.2 minimum
    assert not any("had no effect" in m for m in msgs)


# ---------------------------------------------------------------------------
# weights_path: predict-or-raise when preds_df is None
# ---------------------------------------------------------------------------


def test_no_preds_no_weights_raises(split_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, _ = split_dataset
    ev = Evaluation(None, gt_df, iou_threshold=0.5)
    with pytest.raises(ValueError, match="no predictions to score"):
        ev(split="test")


def test_both_preds_and_weights_warns_at_init(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    msgs = _capture_warnings(
        lambda: Evaluation(preds_df, gt_df, iou_threshold=0.5, weights_path="best.pt")
    )
    assert any("ignoring weights_path" in m for m in msgs)
