"""Calibration work must follow changed images rather than every global score."""

import numpy as np
import pandas as pd
import pytest

from digital_metrics.backends import operating_points
from digital_metrics.calibration import ConfidenceCalibrator
from digital_metrics.engines.backend import BackendEngine

COLS = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


@pytest.mark.parametrize("backend", ["torchmetrics", "ultralytics"])
@pytest.mark.parametrize("mode", ["global", "per_class"])
def test_calibration_matches_each_independent_image_once(
    monkeypatch: pytest.MonkeyPatch, backend: str, mode: str
) -> None:
    if backend == "ultralytics":
        pytest.importorskip("ultralytics")
    size = 40
    gt = pd.DataFrame([[str(i), "cat", 0, 0, 10, 10] for i in range(size)], columns=COLS)
    preds = gt.assign(confidence=np.arange(1, size + 1) / (size + 1))
    original = operating_points.compute_iou_matrix
    original_counts = operating_points._group_counts
    calls = 0
    rematches = 0

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    def counted_matches(*args: object, **kwargs: object) -> object:
        nonlocal rematches
        rematches += 1
        return original_counts(*args, **kwargs)

    monkeypatch.setattr(operating_points, "_group_counts", counted_matches)
    monkeypatch.setattr(operating_points, "compute_iou_matrix", counted)
    actual = operating_points.calibrate_raw(gt, preds, ["cat"], None, mode, backend)
    expected = 1 / (size + 1) if mode == "global" else {"cat": 1 / (size + 1)}
    assert actual == pytest.approx(expected)
    assert calls <= size
    assert rematches == size


def test_missing_backend_summary_has_unavailable_counts() -> None:
    gt = pd.DataFrame([["i", "cat", 0, 0, 10, 10]], columns=COLS)
    engine = BackendEngine(
        backend="torchmetrics",
        classes=["cat", "dog"],
        confidence_optimization="per_class",
        calibrator=ConfidenceCalibrator(
            classes=["cat", "dog"],
            iou_threshold=0.5,
            matching_strategy="greedy",
            confidence_optimization="per_class",
        ),
    )
    absent = engine._adapt({}, gt)["dog"]
    assert not absent.counts_observed
    assert np.isnan(absent.precision_ci_upper) and np.isnan(absent.recall_ci_lower)


def test_coco_float32_coordinates_and_score_order() -> None:
    gt = pd.DataFrame([["i", "cat", 0, 0, 10, 10]], columns=COLS)
    preds = pd.DataFrame([["i", "cat", 0, 0, 20.0000001, 10, 0.5]], columns=COLS + ["confidence"])
    assert operating_points.raw_counts(gt, preds, ["cat"], None, 0.5, "torchmetrics")["cat"] == (
        1,
        0,
        0,
    )
    gt = pd.concat([gt, gt.assign(bbox_x_tl=5, bbox_x_br=15)], ignore_index=True)
    preds = pd.DataFrame(
        [["i", "cat", 0, 0, 15, 10, 0.9], ["i", "cat", 5, 0, 15, 10, 0.900000001]],
        columns=COLS + ["confidence"],
    )
    assert operating_points.raw_counts(gt, preds, ["cat"], None, 0.5, "torchmetrics")["cat"] == (
        1,
        1,
        1,
    )


@pytest.mark.parametrize("backend", ["torchmetrics", "ultralytics"])
@pytest.mark.parametrize("mode", ["global", "per_class"])
def test_incremental_calibration_equals_full_rematching(backend: str, mode: str) -> None:
    from fractions import Fraction

    if backend == "ultralytics":
        pytest.importorskip("ultralytics")
    gt = pd.DataFrame(
        [
            ["a", "cat", 0, 0, 10, 10],
            ["a", "cat", 5, 0, 15, 10],
            ["b", "dog", 0, 0, 10, 10],
            ["empty", None, None, None, None, None],
        ],
        columns=COLS,
    )
    preds = pd.DataFrame(
        [
            ["a", "cat", 0, 0, 15, 10, 0.9],
            ["a", "cat", 5, 0, 15, 10, 0.900000001],
            ["a", "cat", 0, 0, 10, 10, 0.4],
            ["b", "dog", 0, 0, 10, 10, 0.4],
            ["empty", "cat", 0, 0, 10, 10, 0.7],
            ["empty", "dog", 0, 0, 10, 10, 0.9],
        ],
        columns=COLS + ["confidence"],
    )
    classes = ["cat", "dog"]
    candidates = sorted(set(preds.confidence), reverse=True)
    best = {name: (Fraction(-1), 0.0) for name in classes}
    global_best, global_threshold = Fraction(-1), 0.0
    for threshold in candidates:
        counts = operating_points.raw_counts(gt, preds, classes, None, threshold, backend)
        values = {}
        for name, (tp, fp, fn) in counts.items():
            values[name] = Fraction(2 * tp, 2 * tp + fp + fn)
            if values[name] > best[name][0]:
                best[name] = values[name], threshold
        total = sum(values.values())
        if total > global_best:
            global_best, global_threshold = total, threshold
    expected = global_threshold if mode == "global" else {name: best[name][1] for name in classes}
    calibrated = operating_points.calibrate_raw(gt, preds, classes, None, mode, backend)
    if backend != "ultralytics" or mode == "global":
        assert calibrated == expected
    else:
        assert isinstance(calibrated, dict)
        counts = operating_points.raw_counts(gt, preds, classes, None, calibrated, backend)
        realized = sum(Fraction(2 * tp, 2 * tp + fp + fn) for tp, fp, fn in counts.values())
        for name in classes:
            for candidate in set(preds.loc[preds.instance_label == name, "confidence"]) | {
                max(candidates)
            }:
                proposed = operating_points.raw_counts(
                    gt, preds, classes, None, calibrated | {name: candidate}, backend
                )
                assert (
                    sum(Fraction(2 * tp, 2 * tp + fp + fn) for tp, fp, fn in proposed.values())
                    <= realized
                )


def test_coco_float32_xywh_conversion() -> None:
    gt = pd.DataFrame([["i", "cat", -100000000, 0, 1, 10]], columns=COLS)
    preds = pd.DataFrame(
        [["i", "cat", -100000000, 0, -50000000, 10, 0.5]], columns=COLS + ["confidence"]
    )
    assert operating_points.raw_counts(gt, preds, ["cat"], None, 0.5, "torchmetrics")["cat"] == (
        1,
        0,
        0,
    )


def test_thresholded_class_absent_from_gt_retains_observed_fp_counts() -> None:
    gt = pd.DataFrame([["i", "cat", 0, 0, 10, 10]], columns=COLS)
    preds = pd.DataFrame([["i", "dog", 0, 0, 10, 10, 0.5]], columns=COLS + ["confidence"])
    result = operating_points.complete_counts(
        {}, gt, preds, ["cat", "dog"], None, 0.5, "torchmetrics"
    )
    assert (result["dog"].tp, result["dog"].fp, result["dog"].fn) == (0, 1, 0)
    assert np.isnan(result["dog"].ap50)


@pytest.mark.parametrize("size", [20, 40])
def test_mixed_class_coordinate_search_reuses_iou_and_changed_images(
    monkeypatch: pytest.MonkeyPatch, size: int
) -> None:
    pytest.importorskip("ultralytics")
    from ultralytics.utils import metrics

    target_rows = [["1", 0, 0, 20, 10], ["1", 0, 0, 20, 10], ["0", 5, 0, 15, 10]]
    prediction_rows = [
        ["1", 0, 0, 10, 10, 0.3],
        ["1", 5, 0, 15, 10, 0.3],
        ["0", 0, 0, 20, 10, 0.7],
        ["1", 0, 0, 20, 10, 0.9],
        ["1", 0, 0, 10, 10, 0.7],
        ["1", 5, 0, 15, 10, 0.9],
        ["0", 0, 0, 20, 10, 0.7],
        ["0", 0, 0, 20, 10, 0.9],
    ]
    gt = pd.DataFrame([[str(i), *row] for i in range(size) for row in target_rows], columns=COLS)
    preds = pd.DataFrame(
        [
            [str(i), *row[:-1], row[-1] * (0.5 + 0.5 * i / (size + 1))]
            for i in range(size)
            for row in prediction_rows
        ],
        columns=COLS + ["confidence"],
    )
    original_iou, original_counts = metrics.box_iou, operating_points._group_counts
    ious = rematches = 0

    def count_iou(*args: object, **kwargs: object) -> object:
        nonlocal ious
        ious += 1
        return original_iou(*args, **kwargs)

    def count_matches(*args: object, **kwargs: object) -> object:
        nonlocal rematches
        rematches += 1
        return original_counts(*args, **kwargs)

    monkeypatch.setattr(metrics, "box_iou", count_iou)
    monkeypatch.setattr(operating_points, "_group_counts", count_matches)
    result = operating_points.calibrate_raw(gt, preds, ["0", "1"], None, "per_class", "ultralytics")
    assert isinstance(result, dict) and set(result) == {"0", "1"}
    assert ious == size
    assert rematches <= 30 * size
