"""Single entry point for external (library-backed) detection metrics.

:func:`compute_detection_metrics` dispatches to one of two optional backends,
both returning ``dict[str, DetectionMetrics]`` over the same box inputs:

* ``"ultralytics"`` — YOLO-comparable P/R/F1/AP from Ultralytics' own
  ``ap_per_class`` (see :mod:`metrics.ultralytics_metrics`).
* ``"torchmetrics"`` — general COCO mAP from torchmetrics'
  ``MeanAveragePrecision`` (see :mod:`metrics.torchmetrics_metrics`).

Each backend is a heavy optional extra (both pull in ``torch``); the relevant
dependency is imported lazily inside the backend, so importing this module stays
torch-free. Pick one with ``pip install digital-metrics[ultralytics]`` or
``digital-metrics[torchmetrics]``.
"""

from __future__ import annotations

from typing import Literal, get_args

import pandas as pd

from ..types import DetectionMetrics
from .torchmetrics_metrics import compute_torchmetrics_metrics
from .ultralytics_metrics import compute_ultralytics_metrics

Backend = Literal["ultralytics", "torchmetrics"]


def compute_detection_metrics(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    *,
    backend: Backend = "ultralytics",
    classes: list[str] | None = None,
    split_image_names: list[str] | None = None,
) -> dict[str, DetectionMetrics]:
    """Compute per-class detection metrics with the chosen external backend.

    Args:
        gt_df: Ground-truth DataFrame (standard schema), already scoped to the
            split you want to score.
        preds_df: Predictions DataFrame (standard schema). Predictions on images
            outside the split are dropped.
        backend: Which metric library to use. ``"ultralytics"`` (default) gives
            YOLO-comparable numbers; ``"torchmetrics"`` gives general COCO mAP.
        classes: Class vocabulary fixing the label→index mapping. Defaults to the
            sorted union of GT and prediction labels.
        split_image_names: Complete list of image names in the split, including
            empty images (no GT). Predictions on those images are counted as
            false positives. Defaults to the images present in ``gt_df``.

    Returns:
        ``{class_name: DetectionMetrics}`` for every class with at least one GT
        box in the split.

    Raises:
        ValueError: If ``backend`` is not one of the supported values.
        ImportError: If the selected backend's optional dependency is missing.
    """
    if backend == "ultralytics":
        return compute_ultralytics_metrics(
            gt_df, preds_df, classes=classes, split_image_names=split_image_names
        )
    if backend == "torchmetrics":
        return compute_torchmetrics_metrics(
            gt_df, preds_df, classes=classes, split_image_names=split_image_names
        )
    raise ValueError(f"Unknown backend {backend!r}; expected one of {list(get_args(Backend))}.")
