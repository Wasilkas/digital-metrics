"""Tests for the YOLO inference adapter (Evaluation.predict_to_dataframe).

The pure conversion helpers are torch-free and always run; the model-loading path
needs the optional ``ultralytics`` extra, so only the validation and lazy-import
errors are checked here.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from metrics import Evaluation
from metrics.yolo_predict import _PRED_COLUMNS, _detection_rows

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
