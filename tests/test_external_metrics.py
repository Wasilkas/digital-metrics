"""Tests for the single external-metrics entry point (compute_detection_metrics).

The dispatcher itself is import-light; the backend round-trips are skipped unless
the corresponding optional extra (``ultralytics`` / ``torchmetrics``) is present.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from digital_metrics import DetectionMetrics, compute_detection_metrics
from digital_metrics.backends.ultralytics_metrics import _confusion_process_batch
from digital_metrics.validation import drop_na_labels

_BACKENDS = ["ultralytics", "torchmetrics"]


def test_ultralytics_confusion_counts_fp_when_there_are_no_matches() -> None:
    """Non-overlapping predictions remain FPs even when no match exists."""
    matrix = np.zeros((3, 3), dtype=np.int64)

    _confusion_process_batch(
        matrix,
        det_classes=np.array([1]),
        gt_classes=np.array([0]),
        iou=np.array([[0.0]]),
        iou_thres=0.45,
        nc=2,
    )

    assert matrix[2, 0] == 1  # unmatched GT -> FN
    assert matrix[1, 2] == 1  # unmatched detection -> FP
    assert matrix.sum() == 2


def test_unknown_backend_raises_value_error(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    # Wrong backend must fail fast, before any heavy import is attempted.
    with pytest.raises(ValueError, match="Unknown backend"):
        compute_detection_metrics(gt_df, preds_df, backend="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", _BACKENDS)
def test_dispatch_returns_detectionmetrics(
    backend: str,
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    module = "torchmetrics" if backend == "torchmetrics" else "ultralytics"
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"optional backend {module!r} not installed")

    gt_df, preds_df = tiny_dataset
    result = compute_detection_metrics(gt_df, preds_df, backend=backend)  # type: ignore[arg-type]

    assert set(result) == {"class_a", "class_b", "class_c"}
    assert all(isinstance(m, DetectionMetrics) for m in result.values())


def test_drop_na_labels() -> None:
    """Placeholder rows for empty images (NA label) are removed."""
    df = pd.DataFrame([("i1", "class_a"), ("i2", None)], columns=["image_name", "instance_label"])
    assert drop_na_labels(df)["image_name"].tolist() == ["i1"]
    # Nothing to drop: the same object comes back (no needless copy).
    clean = df.iloc[:1]
    assert drop_na_labels(clean) is clean


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_scores_empty_image_rows(
    backend: str,
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A NA-label GT row must not break class sorting / label→index mapping."""
    module = "torchmetrics" if backend == "torchmetrics" else "ultralytics"
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"optional backend {module!r} not installed")

    gt_df, preds_df = tiny_dataset
    empty_image = pd.DataFrame(
        [("img3", None, None, None, None, None, "test")], columns=gt_df.columns
    )
    gt_df = pd.concat([gt_df, empty_image], ignore_index=True)
    result = compute_detection_metrics(gt_df, preds_df, backend=backend)  # type: ignore[arg-type]

    assert set(result) == {"class_a", "class_b", "class_c"}
