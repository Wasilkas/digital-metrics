"""Compare retained detections with actual backend matchers, not AP envelopes."""

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from digital_metrics.backends.operating_points import calibrate_raw, raw_counts

COLS = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def dataset(case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.DataFrame([["positive", "cat", 0, 0, 10, 10]], columns=COLS)
    detections = [["positive", "cat", 0, 0, 10, 10, 0.5]]
    if case == "zero_tp":
        detections = [["positive", "cat", 20, 20, 30, 30, 0.5]]
    elif case == "empty_fp":
        detections.append(["negative", "cat", 0, 0, 10, 10, 0.5])
    elif case == "iou_boundary":
        detections = [["positive", "cat", 0, 0, 20, 10, 0.5]]
    elif case == "float32_iou":
        detections = [["positive", "cat", 0, 0, 20.0000001, 10, 0.5]]
    elif case == "float32_xywh":
        gt = pd.DataFrame([["positive", "cat", -100000000, 0, 1, 10]], columns=COLS)
        detections = [["positive", "cat", -100000000, 0, -50000000, 10, 0.5]]
    elif case == "float32_scores":
        gt = pd.concat([gt, gt.assign(bbox_x_tl=5, bbox_x_br=15)], ignore_index=True)
        detections = [
            ["positive", "cat", 0, 0, 15, 10, 0.9],
            ["positive", "cat", 5, 0, 15, 10, 0.900000001],
        ]
    elif case == "ties":
        gt = pd.concat([gt, gt.assign(bbox_x_tl=5, bbox_x_br=15)], ignore_index=True)
        detections.append(["positive", "cat", 5, 0, 15, 10, 0.5])
    elif case in ("iou_ties", "iou_ties_reversed"):
        gt = pd.concat([gt, gt], ignore_index=True)
        detections.append(["positive", "cat", 0, 0, 10, 10, 0.5])
        detections.append(["positive", "cat", 0, 0, 10, 10, 0.7])
        if case == "iou_ties_reversed":
            detections.reverse()
    elif case == "limit":
        detections = [["positive", "cat", 20, 20, 30, 30, 0.9]] * 100 + detections
    return gt, pd.DataFrame(detections, columns=COLS + ["confidence"])


@pytest.mark.parametrize(
    "case",
    [
        "zero_tp",
        "empty_fp",
        "iou_boundary",
        "ties",
        "iou_ties",
        "iou_ties_reversed",
        "limit",
        "float32_iou",
        "float32_scores",
        "float32_xywh",
    ],
)
def test_ultralytics_actual_validator(case: str) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou

    gt, preds = dataset(case)
    actual_tp = 0
    for image in ["positive", "negative"]:
        g = gt[gt.image_name == image]
        p = preds[(preds.image_name == image) & (preds.confidence >= 0.5)]
        validator = object.__new__(BaseValidator)
        validator.iouv = torch.linspace(0.5, 0.95, 10)
        correct = validator.match_predictions(
            torch.zeros(len(p)),
            torch.zeros(len(g)),
            box_iou(
                torch.tensor(g[COLS[2:]].to_numpy(np.float32)),
                torch.tensor(p[COLS[2:]].to_numpy(np.float32)),
            ),
        )
        actual_tp += int(correct[:, 0].sum())
    result = raw_counts(gt, preds, ["cat"], ["positive", "negative"], 0.5, "ultralytics")
    assert result["cat"] == (actual_tp, len(preds) - actual_tp, len(gt) - actual_tp)


@pytest.mark.parametrize(
    "case",
    [
        "zero_tp",
        "empty_fp",
        "iou_boundary",
        "ties",
        "iou_ties",
        "iou_ties_reversed",
        "limit",
        "float32_iou",
        "float32_scores",
        "float32_xywh",
    ],
)
def test_coco_actual_events(case: str) -> None:
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt, preds = dataset(case)
    # Use the actual TorchMetrics adapter tensor precision, not an idealized
    # float64 COCO input. Retention still compares original confidence values.
    original_gt, original_preds = gt.copy(), preds.copy()
    gt[COLS[2:]] = gt[COLS[2:]].astype(np.float32)
    preds[COLS[2:] + ["confidence"]] = preds[COLS[2:] + ["confidence"]].astype(np.float32)
    images = {"positive": 1, "negative": 2}
    coco = COCO()
    annotations = []
    for i, row in gt.iterrows():
        box = [
            float(row.bbox_x_tl),
            float(row.bbox_y_tl),
            float(np.float32(row.bbox_x_br) - np.float32(row.bbox_x_tl)),
            float(np.float32(row.bbox_y_br) - np.float32(row.bbox_y_tl)),
        ]
        annotations.append(
            dict(
                id=i + 1,
                image_id=images[row.image_name],
                category_id=1,
                bbox=box,
                area=box[2] * box[3],
                iscrowd=0,
            )
        )
    coco.dataset = dict(
        info={},
        images=[dict(id=i) for i in images.values()],
        categories=[dict(id=1, name="cat")],
        annotations=annotations,
    )
    predictions = []
    for _, row in preds[preds.confidence >= 0.5].iterrows():
        predictions.append(
            dict(
                image_id=images[row.image_name],
                category_id=1,
                score=float(row.confidence),
                bbox=[
                    float(row.bbox_x_tl),
                    float(row.bbox_y_tl),
                    float(np.float32(row.bbox_x_br) - np.float32(row.bbox_x_tl)),
                    float(np.float32(row.bbox_y_br) - np.float32(row.bbox_y_tl)),
                ],
            )
        )
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(predictions), "bbox")
        evaluator.params.iouThrs = np.array([0.5])
        evaluator.evaluate()
    events = [e for e in evaluator.evalImgs if e is not None and e["aRng"] == [0, 1e10]]
    tp = sum(int((e["dtMatches"][0] > 0).sum()) for e in events)
    count = sum(len(e["dtIds"]) for e in events)
    assert raw_counts(original_gt, original_preds, ["cat"], list(images), 0.5, "torchmetrics")[
        "cat"
    ] == (
        tp,
        count - tp,
        len(gt) - tp,
    )


def test_calibration_preserves_empty_image_scope_and_ties() -> None:
    gt, preds = dataset("empty_fp")
    preds.loc[1, "confidence"] = 0.9
    threshold = calibrate_raw(
        gt, preds, ["cat"], ["positive", "negative"], "global", "torchmetrics"
    )
    assert threshold == 0.5
    assert raw_counts(gt, preds, ["cat"], ["positive", "negative"], threshold, "torchmetrics")[
        "cat"
    ] == (1, 1, 0)


def mixed_class_ties() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt = pd.DataFrame(
        [
            ["i", "1", 0, 0, 20, 10],
            ["i", "1", 0, 0, 20, 10],
            ["i", "0", 5, 0, 15, 10],
        ],
        columns=COLS,
    )
    preds = pd.DataFrame(
        [
            ["i", "1", 0, 0, 10, 10, 0.5],
            ["i", "1", 5, 0, 15, 10, 0.5],
            ["i", "0", 0, 0, 20, 10, 0.5],
            ["i", "1", 0, 0, 20, 10, 0.5],
            ["i", "1", 0, 0, 10, 10, 0.5],
            ["i", "1", 5, 0, 15, 10, 0.5],
            ["i", "0", 0, 0, 20, 10, 0.5],
            ["i", "0", 0, 0, 20, 10, 0.5],
        ],
        columns=COLS + ["confidence"],
    )
    return gt, preds


def validator_counts(
    gt: pd.DataFrame, preds: pd.DataFrame, threshold: float | dict[str, float]
) -> dict[str, tuple[int, int, int]]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics")
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import box_iou

    thresholds = preds.instance_label.map(threshold) if isinstance(threshold, dict) else threshold
    preds = preds[preds.confidence >= thresholds]
    validator = object.__new__(BaseValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    correct = validator.match_predictions(
        torch.tensor(preds.instance_label.astype(int).to_numpy()),
        torch.tensor(gt.instance_label.astype(int).to_numpy()),
        box_iou(
            torch.tensor(gt[COLS[2:]].to_numpy(np.float32)),
            torch.tensor(preds[COLS[2:]].to_numpy(np.float32)),
        ),
    )[:, 0].numpy()
    return {
        name: (
            int(correct[preds.instance_label.to_numpy() == name].sum()),
            int((preds.instance_label == name).sum())
            - int(correct[preds.instance_label.to_numpy() == name].sum()),
            int((gt.instance_label == name).sum())
            - int(correct[preds.instance_label.to_numpy() == name].sum()),
        )
        for name in ["0", "1"]
    }


@pytest.mark.parametrize("threshold", [0.5, {"0": 0.6, "1": 0.5}, {"0": 0.5, "1": 0.6}])
def test_ultralytics_mixed_class_whole_image_ties(threshold: float | dict[str, float]) -> None:
    gt, preds = mixed_class_ties()
    expected = validator_counts(gt, preds, threshold)
    if threshold == 0.5:
        assert expected["1"] == (1, 4, 1)
    assert raw_counts(gt, preds, ["0", "1"], None, threshold, "ultralytics") == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("mode", ["global", "per_class"])
def test_mixed_class_calibration_matches_whole_image_validator(mode: str, *, reverse: bool) -> None:
    from fractions import Fraction

    gt, preds = mixed_class_ties()
    preds.confidence = [0.9, 0.7, 0.8, 0.5, 0.7, 0.4, 0.9, 0.8]
    if reverse:
        gt, preds = gt.iloc[::-1], preds.iloc[::-1]
    candidates = sorted(set(preds.confidence), reverse=True)
    best = {name: (Fraction(-1), 0.0) for name in ["0", "1"]}
    global_best, global_threshold = Fraction(-1), 0.0
    for threshold in candidates:
        expected = validator_counts(gt, preds, threshold)
        assert raw_counts(gt, preds, ["0", "1"], None, threshold, "ultralytics") == expected
        values = {}
        for name, (tp, fp, fn) in expected.items():
            values[name] = Fraction(2 * tp, 2 * tp + fp + fn)
            if values[name] > best[name][0]:
                best[name] = values[name], threshold
        total = sum(values.values())
        if total > global_best:
            global_best, global_threshold = total, threshold
    calibrated = calibrate_raw(gt, preds, ["0", "1"], None, mode, "ultralytics")
    if mode == "global":
        assert calibrated == global_threshold
    else:
        assert isinstance(calibrated, dict)
        assert raw_counts(
            gt, preds, ["0", "1"], None, calibrated, "ultralytics"
        ) == validator_counts(gt, preds, calibrated)
    assert raw_counts(
        gt, preds, ["0", "1"], None, {"0": 0.8, "1": 0.7}, "ultralytics"
    ) == validator_counts(gt, preds, {"0": 0.8, "1": 0.7})


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("classes", [["0", "1"], ["1", "0"]])
def test_ultralytics_per_class_calibration_uses_realized_dictionary(
    classes: list[str], *, reverse: bool
) -> None:
    from fractions import Fraction

    pytest.importorskip("ultralytics")
    gt, preds = mixed_class_ties()
    preds.confidence = [0.3, 0.3, 0.7, 0.9, 0.7, 0.9, 0.7, 0.9]
    if reverse:
        gt, preds = gt.iloc[::-1], preds.iloc[::-1]
    thresholds = calibrate_raw(gt, preds, classes, None, "per_class", "ultralytics")
    assert isinstance(thresholds, dict)

    def objective(values: dict[str, float]) -> Fraction:
        return sum(
            (
                Fraction(2 * tp, 2 * tp + fp + fn)
                for tp, fp, fn in validator_counts(gt, preds, values).values()
            ),
            Fraction(0),
        )

    actual = objective(thresholds)
    assert actual >= objective({"0": 0.9, "1": 0.9})
    for name in ["0", "1"]:
        candidates = set(preds.loc[preds.instance_label == name, "confidence"]) | {
            float(preds.confidence.max())
        }
        for threshold in candidates:
            proposal = thresholds | {name: threshold}
            proposed = objective(proposal)
            assert proposed <= actual
            if proposed == actual:
                assert threshold <= thresholds[name]
    assert raw_counts(gt, preds, classes, None, thresholds, "ultralytics") == validator_counts(
        gt, preds, thresholds
    )
