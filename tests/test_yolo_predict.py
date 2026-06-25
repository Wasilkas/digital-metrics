"""Tests for the YOLO inference adapter (Evaluation.predict_to_dataframe).

The pure conversion helpers are torch-free and always run; the model-loading path
needs the optional ``ultralytics`` extra, so only the validation and lazy-import
errors are checked here.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from digital_metrics import Evaluation
from digital_metrics.inference.yolo_predict import _PRED_COLUMNS, _detection_rows

_REQUIRED = [
    "image_name",
    "instance_label",
    "confidence",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
]


def test_detection_rows_schema_and_values() -> None:
    xyxy = np.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=float)
    conf = np.array([0.9, 0.5])
    cls = np.array([0, 2])
    names = {0: "a", 1: "b", 2: "c"}

    rows = _detection_rows("/data/imgs/00123_06.jpg", xyxy, conf, cls, names)

    assert len(rows) == 2
    first = rows[0]
    assert first["image_name"] == "00123_06.jpg"  # last path part (filename + ext)
    assert first["instance_label"] == "a"
    assert first["confidence"] == pytest.approx(0.9)
    assert (first["bbox_x_tl"], first["bbox_y_tl"]) == (10.0, 20.0)
    assert (first["bbox_x_br"], first["bbox_y_br"]) == (30.0, 40.0)
    assert "image_path" not in first  # image_path lives on the GT, not predictions
    assert rows[1]["instance_label"] == "c"  # cls index 2 → names[2]

    df = pd.DataFrame(rows, columns=_PRED_COLUMNS)
    assert list(df.columns) == _PRED_COLUMNS
    assert set(_REQUIRED) == set(df.columns)


def test_detection_rows_image_name_modes() -> None:
    box = np.zeros((1, 4))
    conf = np.array([0.5])
    cls = np.array([0])
    names = {0: "x"}

    stem = _detection_rows("/a/b/img_01.jpg", box, conf, cls, names, image_name_mode="stem")
    path = _detection_rows("/a/b/img_01.jpg", box, conf, cls, names, image_name_mode="path")

    assert stem[0]["image_name"] == "img_01"
    assert path[0]["image_name"] == "/a/b/img_01.jpg"


def test_predict_requires_image_path_column(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset  # split_dataset has no image_path column
    ev = Evaluation(preds_df, gt_df)
    with pytest.raises(ValueError, match="image_path"):
        ev.predict_to_dataframe("weights.pt")


def test_predict_none_preds_starts_empty(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, _ = split_dataset
    ev = Evaluation(None, gt_df)  # construct without predictions
    assert list(ev.preds_df.columns)  # placeholder has the required columns
    assert len(ev.preds_df) == 0


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics installed — model path would run instead",
)
def test_missing_ultralytics_raises_importerror(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    gt_df = gt_df.assign(image_path="/imgs/" + gt_df["image_name"] + ".jpg")
    ev = Evaluation(preds_df, gt_df)
    with pytest.raises(ImportError, match="ultralytics"):
        ev.predict_to_dataframe("weights.pt")


@pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is not None,
    reason="ultralytics installed — model path would run instead",
)
def test_weights_path_triggers_prediction_on_call(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # preds_df=None + weights_path → calling the evaluation runs inference, which
    # here fails on the missing extra (proving the auto-predict path is reached).
    gt_df, _ = split_dataset
    gt_df = gt_df.assign(image_path="/imgs/" + gt_df["image_name"] + ".jpg")
    ev = Evaluation(None, gt_df, weights_path="weights.pt")
    with pytest.raises(ImportError, match="ultralytics"):
        ev(split="test")


def test_predict_filters_to_list_of_splits(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    # A split with no rows yields no image paths — proving the split filter ran
    # before any model load (so it works without the ultralytics extra).
    gt_df, preds_df = split_dataset
    gt_df = gt_df.assign(image_path="/imgs/" + gt_df["image_name"] + ".jpg")
    ev = Evaluation(preds_df, gt_df)
    with pytest.raises(ValueError, match="No 'image_path' values"):
        ev.predict_to_dataframe("weights.pt", split=["train"])


def test_predict_split_requires_split_column(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    gt_df = gt_df.drop(columns="split").assign(image_path="/imgs/" + gt_df["image_name"] + ".jpg")
    ev = Evaluation(preds_df, gt_df)
    with pytest.raises(ValueError, match="requires a 'split' column"):
        ev.predict_to_dataframe("weights.pt", split="test")


def test_splits_to_predict_unions_eval_and_calibration(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    ev = Evaluation(preds_df, gt_df)
    assert ev._splits_to_predict("test", "val") == ["test", "val"]
    assert ev._splits_to_predict("test", None) == ["test"]
    assert ev._splits_to_predict("test", "test") == ["test"]  # no duplicate
    assert ev._splits_to_predict("all", "val") is None  # "all" already spans splits


def test_auto_predict_targets_eval_and_calibration_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The weights flow must predict on exactly the eval + calibration splits,
    # not the whole split_df (here: skip 'train'). Stub out the model call and
    # capture which image paths it was handed.
    import digital_metrics.evaluation as evaluation_module

    gt_df = pd.DataFrame(
        [
            ("train1", "class_a", 0, 0, 10, 10, "train"),
            ("val1", "class_a", 0, 0, 10, 10, "val"),
            ("test1", "class_a", 0, 0, 10, 10, "test"),
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "split",
        ],
    ).assign(image_path=lambda d: "/imgs/" + d["image_name"] + ".jpg")

    captured: dict[str, list[str]] = {}

    def fake_predict_on_images(weights: str, image_paths: list[str], **_: object) -> pd.DataFrame:
        captured["paths"] = list(image_paths)
        return pd.DataFrame(columns=_PRED_COLUMNS)  # no detections; pipeline still runs

    monkeypatch.setattr(evaluation_module, "predict_on_images", fake_predict_on_images)

    ev = Evaluation(None, gt_df, weights_path="weights.pt")
    ev(split="test", calibration_split="val")

    assert set(captured["paths"]) == {"/imgs/val1.jpg", "/imgs/test1.jpg"}


def test_predict_kwargs_forwarded_to_model(monkeypatch: pytest.MonkeyPatch) -> None:
    # predict_kwargs on the constructor must reach Ultralytics' model.predict via
    # the auto-predict (weights) flow.
    import digital_metrics.evaluation as evaluation_module

    gt_df = pd.DataFrame(
        [("test1", "class_a", 0, 0, 10, 10, "test")],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "split",
        ],
    ).assign(image_path="/imgs/test1.jpg")

    captured: dict[str, object] = {}

    def fake_predict_on_images(
        weights: str, image_paths: list[str], **kwargs: object
    ) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(columns=_PRED_COLUMNS)

    monkeypatch.setattr(evaluation_module, "predict_on_images", fake_predict_on_images)

    ev = Evaluation(
        None,
        gt_df,
        weights_path="weights.pt",
        predict_kwargs={"imgsz": 1280, "half": True, "augment": True},
    )
    ev(split="test")

    assert captured["imgsz"] == 1280  # named param, threaded through
    assert captured["half"] is True  # extra model kwarg
    assert captured["augment"] is True
