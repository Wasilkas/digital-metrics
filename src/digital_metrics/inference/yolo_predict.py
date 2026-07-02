"""YOLO inference → predictions in the standard schema (image-path driven).

Backs :meth:`metrics.Evaluation.predict_to_dataframe`. Given Ultralytics weights
and a list of image paths (taken from the ground-truth DataFrame's ``image_path``
column), it runs the model and returns predictions as ``image_name`` /
``instance_label`` / ``confidence`` / ``bbox_x_tl/y_tl/x_br/y_br`` rows.
``image_name`` is the last part of the path (``Path(image_path).name``), so it
joins back to the ground-truth ``image_name``.

``ultralytics`` (which pulls in ``torch``) is the optional extra, imported lazily,
so the core install stays torch-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy.typing as npt
import pandas as pd
from loguru import logger

ImageNameMode = Literal["name", "stem", "path"]

# The seven columns Evaluation requires (image_path stays on the GT, not here).
_PRED_COLUMNS = [
    "image_name",
    "instance_label",
    "confidence",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
]

_INSTALL_HINT = (
    "Evaluation.predict_to_dataframe requires the optional 'ultralytics' dependency "
    "(which pulls in torch). Install it with:\n"
    "    pip install digital-metrics[ultralytics]\n"
    "or, with uv:\n"
    "    uv pip install ultralytics"
)


def _image_id(path: str, mode: ImageNameMode) -> str:
    """Derive ``image_name`` from an image's path."""
    if mode == "stem":
        return Path(path).stem
    if mode == "path":
        return str(path)
    return Path(path).name  # "name": last path part, filename with extension (default)


def _detection_rows(
    path: str,
    xyxy: npt.NDArray[Any],
    conf: npt.NDArray[Any],
    cls: npt.NDArray[Any],
    names: dict[int, str],
    image_name_mode: ImageNameMode = "name",
) -> list[dict[str, Any]]:
    """Convert one image's YOLO detections to schema rows (pure, torch-free)."""
    name = _image_id(path, image_name_mode)
    rows: list[dict[str, Any]] = []
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
        rows.append(
            {
                "image_name": name,
                "instance_label": names[int(k)],
                "confidence": float(c),
                "bbox_x_tl": float(x1),
                "bbox_y_tl": float(y1),
                "bbox_x_br": float(x2),
                "bbox_y_br": float(y2),
            }
        )
    return rows


def predict_on_images(
    weights: str | Path,
    image_paths: list[str],
    *,
    conf: float = 0.001,
    iou: float = 0.7,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    image_name: ImageNameMode = "name",
    **model_kwargs: Any,
) -> pd.DataFrame:
    """Run a YOLO model over ``image_paths`` and return predictions in our schema.

    Inference is issued in chunks of ``batch`` images (one ``model.predict`` call
    per chunk) rather than one streamed call over the whole list. Ultralytics'
    predictor retains per-image GPU tensors for the lifetime of a single
    ``predict`` call, so streaming the entire list grows peak VRAM roughly linearly
    with the image count and OOMs on large sets; chunking bounds peak VRAM to
    ``batch`` images regardless of how many are scored. ``batch`` is therefore the
    knob to turn when a run runs out of GPU memory (together with ``imgsz`` and
    ``half=True``).

    Args:
        weights: Path to Ultralytics model weights (``.pt``).
        image_paths: Image files to run inference on (typically the unique
            ``image_path`` values from the ground-truth DataFrame).
        conf: Confidence threshold for inference. Defaults to ``0.001`` (YOLO val
            default) so the full precision-recall curve is available downstream.
        iou: NMS IoU threshold for inference (YOLO val default ``0.7``).
        imgsz: Inference image size.
        batch: Number of images per ``model.predict`` call. Bounds peak GPU memory
            (peak ≈ ``batch`` × per-image cost); lower it to fit a smaller card,
            raise it for throughput. Must be ``>= 1``.
        device: Torch device (e.g. ``"0"``, ``"cpu"``); ``None`` lets Ultralytics
            choose.
        image_name: How to fill ``image_name`` — ``"name"`` (filename with
            extension, the default and this project's convention), ``"stem"`` (no
            extension) or ``"path"`` (full path).
        **model_kwargs: Extra keyword arguments forwarded verbatim to
            ``model.predict`` (e.g. ``half``, ``augment``, ``agnostic_nms``,
            ``max_det``, ``classes``, ``retina_masks``). ``source``, ``stream``,
            ``batch`` and ``verbose`` are managed here and must not be passed.

    Returns:
        DataFrame with the columns in ``_PRED_COLUMNS``; empty (with columns) when
        the model detects nothing.

    Raises:
        ImportError: If the optional ``ultralytics`` dependency is not installed.
        ValueError: If ``batch < 1``.
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc

    paths = [str(p) for p in image_paths]
    model = YOLO(str(weights))
    names: dict[int, str] = model.names  # class index → name, as the model emits cls
    logger.info(
        f"Predicting on {len(paths)} images "
        f"(conf={conf}, iou={iou}, imgsz={imgsz}, batch={batch})..."
    )

    rows: list[dict[str, Any]] = []
    n_images = 0
    # Chunk the source list: peak VRAM stays bounded to ``batch`` images because
    # each ``predict`` call releases its retained tensors when it finishes.
    for start in range(0, len(paths), batch):
        chunk = paths[start : start + batch]
        results = model.predict(
            source=chunk,
            stream=True,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
            **model_kwargs,
        )
        # Ultralytics yields results in input order but rewrites ``r.path`` to
        # generic names for a list source, so name from the original path instead
        # (strict zip surfaces any count mismatch rather than silently misalign).
        for path, r in zip(chunk, results, strict=True):
            n_images += 1
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            rows.extend(
                _detection_rows(
                    path,
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    names,
                    image_name,
                )
            )

    logger.info(f"Predicted {len(rows)} boxes over {n_images} images.")
    return pd.DataFrame(rows, columns=_PRED_COLUMNS)
