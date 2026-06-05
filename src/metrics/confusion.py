from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix

from .types import PredictMatch


def get_confusion_matrix(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
) -> tuple[npt.NDArray[np.int64], list[str]]:
    """Build a confusion matrix from match records.

    Args:
        matches: Dict mapping class name → list of PredictMatch objects.
        classes: Ordered list of class names (background appended automatically).

    Returns:
        (cm, class_labels) where cm has shape (n+1, n+1) including background.
    """
    true_labels: list[str] = []
    pred_labels: list[str] = []

    for c in classes:
        for match in matches.get(c, []):
            true_labels.append(match.gt_label)
            pred_labels.append(match.pred_label)

    class_labels = list(classes) + ["background"]
    cm: npt.NDArray[np.int64] = confusion_matrix(true_labels, pred_labels, labels=class_labels)
    return cm, class_labels


def _get_boxes_to_save(
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    gt_index: int,
    pred_index: int,
) -> list[dict[str, Any]]:
    """Build annotation-audit records for a single match."""
    fields = ["instance_label", "bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br", "image_name"]
    res_list: list[dict[str, Any]] = []

    if pred_index != -1:
        pred_to_save: dict[str, Any] = {
            str(k): v
            for k, v in preds_df.loc[pred_index][fields + ["confidence"]].to_dict().items()
        }
        predict_type = "fp" if gt_index != -1 else "bg"
        pred_to_save["type"] = f"predict_{predict_type}"
        res_list.append(pred_to_save)

    if gt_index != -1:
        gt_to_save: dict[str, Any] = {
            str(k): v for k, v in gt_df.loc[gt_index][fields].to_dict().items()
        }
        gt_to_save["type"] = "gt" if pred_index != -1 else "fn"
        gt_to_save["confidence"] = 1
        res_list.append(gt_to_save)

    return res_list


def get_confusions(
    matches: dict[str, list[PredictMatch]],
    class_labels: list[str],
    preds_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    main_class: str,
    subclasses: list[str],
) -> pd.DataFrame:
    """Find annotation errors / confusions between main_class and subclasses.

    Args:
        matches: Filtered match records.
        class_labels: Full label list including background.
        preds_df: Predictions DataFrame.
        gt_df: Ground-truth DataFrame.
        main_class: Primary class to audit.
        subclasses: Other classes whose interactions with main_class are checked.

    Returns:
        DataFrame with box records for visual inspection.
    """
    for clss in subclasses:
        if clss not in class_labels:
            raise ValueError(f"subclass {clss!r} not in dataset classes")

    boxes_to_check: list[dict[str, Any]] = []

    for match in matches.get(main_class, []):
        classes_with_fn = list(subclasses) + [main_class]
        if match.gt_label in classes_with_fn and match.type in ("FP", "FN"):
            boxes_to_check.extend(
                _get_boxes_to_save(preds_df, gt_df, match.gt_index, match.pred_index)
            )

    for sub in subclasses:
        for match in matches.get(sub, []):
            if match.gt_label == main_class:
                boxes_to_check.extend(
                    _get_boxes_to_save(preds_df, gt_df, match.gt_index, match.pred_index)
                )

    return pd.DataFrame(boxes_to_check)
