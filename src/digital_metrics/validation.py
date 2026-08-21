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
