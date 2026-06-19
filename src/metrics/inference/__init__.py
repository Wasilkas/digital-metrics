"""Optional YOLO inference: run Ultralytics weights into the prediction schema."""

from .yolo_predict import ImageNameMode, predict_on_images

__all__ = ["ImageNameMode", "predict_on_images"]
