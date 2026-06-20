"""Input validation shared by the orchestrator and the scoring engines.

A tiny foundation module (depends only on pandas) so both
:class:`~metrics.evaluation.Evaluation` and the engines in ``engines/`` can
validate ground-truth / prediction DataFrames without importing each other.
"""

from __future__ import annotations

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


def validate_dataframes(preds_df: pd.DataFrame, gt_df: pd.DataFrame, classes: list[str]) -> None:
    """Validate the prediction and ground-truth DataFrames before scoring.

    Args:
        preds_df: Predictions DataFrame.
        gt_df: Ground-truth DataFrame for the evaluated split.
        classes: Known ground-truth class vocabulary.

    Raises:
        ValueError: If a required column is missing, the predictions
            ``confidence`` column has ``NA`` values, or a prediction label is
            absent from ``classes``.
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

    # Predictions must only use classes present in the ground-truth vocabulary.
    gt_classes = set(classes)
    pred_classes = set(preds_df["instance_label"].dropna().unique())
    unknown = pred_classes - gt_classes
    if unknown:
        raise ValueError(
            f"Prediction labels not present in ground truth: {sorted(unknown)}. "
            f"Known ground-truth classes: {sorted(gt_classes)}."
        )
