"""Optional, torch-backed metric backends (Ultralytics / torchmetrics).

Each backend is a heavy optional extra; ``torch`` is imported lazily inside the
backend functions, so importing this package stays torch-free.
"""

from .external import Backend, compute_detection_metrics
from .torchmetrics_metrics import compute_torchmetrics_metrics, find_torchmetrics_confidence
from .ultralytics_metrics import (
    YoloMetrics,
    compute_ultralytics_confusion_matrix,
    compute_ultralytics_metrics,
    find_ultralytics_confidence,
)

__all__ = [
    "Backend",
    "YoloMetrics",
    "compute_detection_metrics",
    "compute_torchmetrics_metrics",
    "compute_ultralytics_confusion_matrix",
    "compute_ultralytics_metrics",
    "find_torchmetrics_confidence",
    "find_ultralytics_confidence",
]
