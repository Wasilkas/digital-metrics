"""Input validation shared by the orchestrator and the scoring engines.

A tiny foundation module (depends only on pandas) so both
:class:`~metrics.evaluation.Evaluation` and the engines in ``engines/`` can
validate ground-truth / prediction DataFrames without importing each other.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLS_GT = {
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
}
REQUIRED_COLS_PREDS = {
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
    "confidence",
}

BBOX_COLS = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]


def validate_option(name: str, value: str, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}.")


def validate_iou(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("iou_threshold must be finite and in [0, 1].")


def normalize_image_ids(*frames: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    """Copy frames and normalize image IDs; reject lossy collisions across inputs."""
    originals: dict[str, object] = {}
    result: list[pd.DataFrame] = []
    for frame in frames:
        for value in frame["image_name"].unique():
            if pd.isna(value):
                raise ValueError("image_name must not contain missing values.")
            key = str(value)
            if key in originals and value != originals[key]:
                raise ValueError(f"image_name normalization collision for {key!r}.")
            originals[key] = value
        copy = frame.copy()
        copy["image_name"] = copy["image_name"].map(str)
        result.append(copy)
    return tuple(result)


def validate_split(frame: pd.DataFrame, split: str) -> None:
    if split == "all":
        return
    if "split" not in frame:
        raise ValueError(f"split={split!r} requires a 'split' column.")
    if split not in set(frame["split"].dropna()):
        raise ValueError(f"Requested split {split!r} is unavailable.")


def validate_split_ownership(frame: pd.DataFrame) -> None:
    if "split" in frame:
        sizes = frame.groupby("image_name")["split"].nunique(dropna=False)
        shared = sizes[sizes > 1].index.tolist()
        if shared:
            raise ValueError(f"Each image_name must belong to one split; shared images: {shared}.")


def drop_na_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose ``instance_label`` is ``NA``.

    Empty images carry a placeholder ground-truth row with a ``NA`` label purely
    to keep the image in scope. It names no class, so it must not reach code that
    sorts, indexes or maps labels. Scope the image set *before* calling this.

    Args:
        df: Ground-truth or predictions DataFrame.

    Returns:
        The DataFrame without the ``NA``-label rows (the original object when
        there are none).
    """
    mask = df["instance_label"].notna()
    return df if bool(mask.all()) else df[mask]


def validate_dataframes(preds_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
    """Validate the prediction and ground-truth DataFrames before scoring.

    Args:
        preds_df: Predictions DataFrame.
        gt_df: Ground-truth DataFrame for the evaluated split.

    Predictions whose label is absent from the ground-truth vocabulary are not
        checked here: :class:`~metrics.evaluation.Evaluation` warns about and
        drops them before scoring (see ``_drop_unknown_pred_classes``).

    Raises:
        ValueError: If a required column is missing or the predictions
            ``confidence`` column has ``NA`` values.
    """
    missing_gt = REQUIRED_COLS_GT - set(gt_df.columns)
    if missing_gt:
        raise ValueError(f"Ground-truth DataFrame is missing columns: {sorted(missing_gt)}")
    missing_preds = REQUIRED_COLS_PREDS - set(preds_df.columns)
    if missing_preds:
        raise ValueError(f"Predictions DataFrame is missing columns: {sorted(missing_preds)}")

    na_conf = int(preds_df["confidence"].isna().sum())
    if na_conf:
        raise ValueError(
            f"Predictions 'confidence' column contains {na_conf} NA value(s); "
            "every prediction must have a numeric confidence."
        )
    for name, frame in [("Ground-truth", gt_df), ("Predictions", preds_df)]:
        if (
            not frame.index.is_unique
            or -1 in frame.index
            or any(bool(np.asarray(pd.isna(label)).any()) for label in frame.index)
        ):
            raise ValueError(f"{name} row labels must be unique, nonmissing and distinct from -1.")
        labels = frame["instance_label"]
        if (labels == "background").any():
            raise ValueError("'background' is a reserved class label for unmatched detections.")
        boxes = frame[BBOX_COLS]
        placeholders = labels.isna() & boxes.isna().all(axis=1)
        if name == "Predictions" and labels.isna().any():
            raise ValueError("Predictions instance_label must not be missing.")
        valid = frame.loc[~placeholders]
        if valid["instance_label"].isna().any():
            raise ValueError(f"{name} unlabeled rows must have fully empty boxes.")
        try:
            coordinates = valid[BBOX_COLS].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} boxes must have numeric coordinates.") from exc
        if (coordinates[:, 2:] <= coordinates[:, :2]).any():
            raise ValueError(f"{name} boxes must have ordered corners and positive area.")
        if not np.isfinite(coordinates).all():
            raise ValueError(f"{name} boxes must have complete finite coordinates.")
    try:
        confidences = preds_df["confidence"].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions confidence must be numeric and finite.") from exc
    if not np.isfinite(confidences).all() or ((confidences < 0) | (confidences > 1)).any():
        raise ValueError("Predictions confidence must be numeric and finite.")
