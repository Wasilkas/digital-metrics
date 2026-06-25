import pandas as pd
import pytest
from loguru import logger

from digital_metrics.calibration import ConfidenceCalibrator
from digital_metrics.matching import match_boxes

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


def _calibrator(classes: list[str], optimization: str = "per_class") -> ConfidenceCalibrator:
    return ConfidenceCalibrator(
        classes=classes,
        iou_threshold=0.5,
        matching_strategy="greedy",
        confidence_optimization=optimization,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# validate_calibration_gt
# ---------------------------------------------------------------------------


def test_validate_calibration_gt_returns_split_rows(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, _ = split_dataset
    eval_gt = gt_df[gt_df["split"] == "test"]
    cal = _calibrator(["class_a", "class_b"])

    cal_gt = cal.validate_calibration_gt(gt_df, "val", eval_gt)
    assert set(cal_gt["split"]) == {"val"}
    assert set(cal_gt["image_name"]) == {"img1", "img2"}


def test_validate_calibration_gt_no_split_column_raises(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, _ = tiny_dataset
    no_split = gt_df.drop(columns=["split"])
    cal = _calibrator(["class_a", "class_b", "class_c"])
    with pytest.raises(ValueError, match="'split' column"):
        cal.validate_calibration_gt(no_split, "val", no_split)


def test_validate_calibration_gt_missing_split_raises(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, _ = split_dataset
    eval_gt = gt_df[gt_df["split"] == "test"]
    cal = _calibrator(["class_a", "class_b"])
    with pytest.raises(ValueError, match="nonexistent"):
        cal.validate_calibration_gt(gt_df, "nonexistent", eval_gt)


def test_validate_calibration_gt_overlap_raises(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """An image_name shared between calibration and eval splits is a leak."""
    gt_df, _ = split_dataset
    leaked = gt_df[gt_df["image_name"] == "img3"].copy()
    leaked["split"] = "val"
    leaked_gt = pd.concat([gt_df, leaked], ignore_index=True)
    eval_gt = leaked_gt[leaked_gt["split"] == "test"]

    cal = _calibrator(["class_a", "class_b"])
    with pytest.raises(ValueError, match="shares"):
        cal.validate_calibration_gt(leaked_gt, "val", eval_gt)


# ---------------------------------------------------------------------------
# find_from_matches: per-class vs global dispatch
# ---------------------------------------------------------------------------


def test_find_from_matches_global_shares_one_threshold(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    val_gt = gt_df[gt_df["split"] == "val"]
    matches = match_boxes(
        val_gt,
        preds_df,
        0.5,
        strategy="greedy",
        split_image_names=val_gt["image_name"].unique().tolist(),
    )

    classes = ["class_a", "class_b"]
    global_thr = _calibrator(classes, "global").find_from_matches(matches)
    assert len(set(global_thr.values())) == 1  # one shared threshold for all classes
    assert set(global_thr.keys()) == set(classes)


def test_find_from_matches_per_class_keys(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    val_gt = gt_df[gt_df["split"] == "val"]
    matches = match_boxes(
        val_gt,
        preds_df,
        0.5,
        strategy="greedy",
        split_image_names=val_gt["image_name"].unique().tolist(),
    )

    classes = ["class_a", "class_b"]
    thr = _calibrator(classes, "per_class").find_from_matches(matches)
    assert set(thr.keys()) == set(classes)
    assert all(isinstance(v, float) for v in thr.values())


def test_find_from_matches_unoptimized_warns() -> None:
    """Perfect matches → optimal threshold is the floor → 'had no effect' warning."""
    gt = pd.DataFrame(
        [("i1", "a", 0, 0, 10, 10, "test"), ("i2", "a", 0, 0, 10, 10, "test")],
        columns=_GT_COLS,
    )
    preds = pd.DataFrame(
        [("i1", "a", 0, 0, 10, 10, 0.9), ("i2", "a", 0, 0, 10, 10, 0.4)],
        columns=_PRED_COLS,
    )
    matches = match_boxes(gt, preds, 0.5, strategy="greedy", split_image_names=["i1", "i2"])
    cal = _calibrator(["a"], "per_class")
    msgs = _capture_warnings(lambda: cal.find_from_matches(matches))
    assert any("had no effect" in m for m in msgs)


# ---------------------------------------------------------------------------
# calibrate_native: validate → match → find
# ---------------------------------------------------------------------------


def test_calibrate_native_matches_in_sample_find(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """calibrate_native on 'val' equals find_from_matches on the val matches."""
    gt_df, preds_df = split_dataset
    val_gt = gt_df[gt_df["split"] == "val"]
    test_gt = gt_df[gt_df["split"] == "test"]
    classes = ["class_a", "class_b"]
    cal = _calibrator(classes)

    out_of_sample = cal.calibrate_native(gt_df, "val", test_gt, preds_df)

    val_matches = match_boxes(
        val_gt,
        preds_df,
        0.5,
        strategy="greedy",
        split_image_names=val_gt["image_name"].unique().tolist(),
    )
    in_sample = cal.find_from_matches(val_matches)

    for c in classes:
        assert out_of_sample[c] == pytest.approx(in_sample[c])
