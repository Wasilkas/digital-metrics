import pandas as pd
import pytest

from metrics.preprocess import PredictionPreprocessor

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


# ---------------------------------------------------------------------------
# enabled
# ---------------------------------------------------------------------------


def test_enabled_false_when_no_thresholds() -> None:
    assert PredictionPreprocessor().enabled is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"conf_threshold": 0.5},
        {"nms_containment_threshold": 0.8},
        {"nms_iou_threshold": 0.5},
    ],
)
def test_enabled_true_when_any_threshold(kwargs: dict[str, float]) -> None:
    assert PredictionPreprocessor(**kwargs).enabled is True


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def test_process_noop_returns_frame_unchanged() -> None:
    df = _preds(("img1", "cat", 0, 0, 10, 10, 0.3))
    result = PredictionPreprocessor().process(df)
    assert result is df  # disabled → exact same object, no copy


def test_process_confidence_filter() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 10, 10, 0.9),
        ("img1", "cat", 0, 0, 10, 10, 0.3),
    )
    result = PredictionPreprocessor(conf_threshold=0.5).process(df)
    assert len(result) == 1
    assert result.iloc[0]["confidence"] == pytest.approx(0.9)


def test_process_nms_containment_suppresses_inner_box() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "cat", 10, 10, 50, 50, 0.7),  # inside the larger box
    )
    result = PredictionPreprocessor(nms_containment_threshold=0.8).process(df)
    assert len(result) == 1
    assert result.iloc[0]["confidence"] == pytest.approx(0.9)


def test_process_nms_cross_class_iou_suppresses_lower_conf() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),
        ("img1", "dog", 5, 5, 105, 105, 0.7),  # high overlap, lower conf
    )
    result = PredictionPreprocessor(nms_iou_threshold=0.5).process(df)
    assert len(result) == 1
    assert result.iloc[0]["instance_label"] == "cat"


def test_process_conf_and_nms_combined() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 100, 100, 0.9),  # kept
        ("img1", "cat", 10, 10, 50, 50, 0.7),  # NMS'd (inside larger)
        ("img1", "cat", 0, 0, 100, 100, 0.1),  # conf-filtered
    )
    result = PredictionPreprocessor(conf_threshold=0.5, nms_containment_threshold=0.8).process(df)
    assert len(result) == 1
    assert result.iloc[0]["confidence"] == pytest.approx(0.9)


def test_process_does_not_mutate_input() -> None:
    df = _preds(
        ("img1", "cat", 0, 0, 10, 10, 0.9),
        ("img1", "cat", 0, 0, 10, 10, 0.3),
    )
    before = len(df)
    PredictionPreprocessor(conf_threshold=0.5).process(df)
    assert len(df) == before  # original frame untouched
