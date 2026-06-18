"""Tests for the single external-metrics entry point (compute_detection_metrics).

The dispatcher itself is import-light; the backend round-trips are skipped unless
the corresponding optional extra (``ultralytics`` / ``torchmetrics``) is present.
"""

import importlib.util

import pandas as pd
import pytest

from metrics import DetectionMetrics, compute_detection_metrics

_BACKENDS = ["ultralytics", "torchmetrics"]


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
