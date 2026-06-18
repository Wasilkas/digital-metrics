"""Tests for the optional Ultralytics-backed metrics path.

Skipped unless the ``ultralytics`` extra is installed. Run with::

    uv run --with ultralytics pytest tests/test_ultralytics_metrics.py
"""

import pandas as pd
import pytest

pytest.importorskip("ultralytics")

from metrics import YoloMetrics, compute_ultralytics_metrics  # noqa: E402

_GT_COLS = ["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
_PRED_COLS = [
    "image_name",
    "instance_label",
    "confidence",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
]


def test_returns_yolometrics_for_each_gt_class(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    result = compute_ultralytics_metrics(gt_df, preds_df)

    assert set(result) == {"class_a", "class_b", "class_c"}
    for m in result.values():
        assert isinstance(m, YoloMetrics)
        for v in (m.precision, m.recall, m.f1, m.ap50, m.ap75, m.ap50_95):
            assert 0.0 <= v <= 1.0


def test_perfect_detector_scores_one() -> None:
    gt_df = pd.DataFrame(
        [("img1", "class_x", 0, 0, 100, 100)],
        columns=_GT_COLS,
    )
    preds_df = pd.DataFrame(
        [("img1", "class_x", 0.9, 0, 0, 100, 100)],
        columns=_PRED_COLS,
    )
    result = compute_ultralytics_metrics(gt_df, preds_df)

    assert set(result) == {"class_x"}
    m = result["class_x"]
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    # A single detection cannot reach AP 1.0 under Ultralytics' 101-point
    # interpolation (it tops out at ~0.995); just assert it is essentially full.
    assert m.ap50 == pytest.approx(1.0, abs=0.01)
    assert m.ap50_95 == pytest.approx(1.0, abs=0.01)


def test_no_predictions_returns_zeros() -> None:
    gt_df = pd.DataFrame(
        [("img1", "class_x", 0, 0, 100, 100)],
        columns=_GT_COLS,
    )
    preds_df = pd.DataFrame([], columns=_PRED_COLS)
    result = compute_ultralytics_metrics(gt_df, preds_df)

    assert set(result) == {"class_x"}
    m = result["class_x"]
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.f1 == 0.0
    assert m.ap50 == 0.0
