"""Predictions preprocessing as a single configurable collaborator.

:class:`PredictionPreprocessor` bundles the optional confidence filter and the
custom NMS into one object that turns a predictions DataFrame into a filtered
copy. :class:`~metrics.evaluation.Evaluation` builds one from its constructor
thresholds and applies it to ``preds_df`` (the raw predictions are left
untouched). The class is pure — it reads a frame and returns a new one — so it
is easy to test in isolation.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from .nms import apply_nms, filter_by_confidence

# Sentinel passed to apply_nms to disable a suppression type (threshold > 1 can
# never be reached, so nothing is suppressed on that axis).
_NMS_DISABLED = 1.01


class PredictionPreprocessor:
    """Confidence filtering and/or custom NMS for a predictions DataFrame."""

    def __init__(
        self,
        *,
        conf_threshold: float | None = None,
        nms_containment_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
    ) -> None:
        """Configure the preprocessor.

        Args:
            conf_threshold: Drop predictions with ``confidence < threshold``.
                ``None`` disables confidence filtering.
            nms_containment_threshold: Same-class containment suppression — the
                lower-confidence box is removed when
                ``intersection / min(area_a, area_b) >= threshold``. ``None``
                disables same-class containment suppression.
            nms_iou_threshold: Cross-class IoU suppression — the lower-confidence
                box is removed when two different-class boxes have
                ``IoU >= threshold``. ``None`` disables cross-class NMS.
        """
        self._conf_threshold = conf_threshold
        self._nms_containment_threshold = nms_containment_threshold
        self._nms_iou_threshold = nms_iou_threshold

    @property
    def enabled(self) -> bool:
        """True when any confidence/NMS threshold is configured."""
        return (
            self._conf_threshold is not None
            or self._nms_containment_threshold is not None
            or self._nms_iou_threshold is not None
        )

    def process(self, preds_df: pd.DataFrame) -> pd.DataFrame:
        """Return ``preds_df`` after confidence filtering and/or custom NMS.

        When no threshold is configured this is a no-op and the frame is returned
        unchanged. Each suppression step logs how many rows it removed.
        """
        result = preds_df
        if self._conf_threshold is not None:
            n_before = len(result)
            result = filter_by_confidence(result, self._conf_threshold)
            logger.info(
                f"Predictions confidence filtering removed "
                f"{n_before - len(result)} rows "
                f"(threshold={self._conf_threshold})."
            )

        cont = self._nms_containment_threshold
        iou = self._nms_iou_threshold
        if cont is not None or iou is not None:
            n_before = len(result)
            result = apply_nms(
                result,
                same_class_containment_threshold=cont if cont is not None else _NMS_DISABLED,
                cross_class_iou_threshold=iou if iou is not None else _NMS_DISABLED,
            )
            logger.info(
                f"Predictions NMS removed {n_before - len(result)} rows "
                f"(containment={cont}, iou={iou})."
            )
        return result
