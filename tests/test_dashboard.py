import os
from pathlib import Path

import pandas as pd

from metrics import Evaluation


def _simple_evaluation() -> Evaluation:
    gt_df = pd.DataFrame(
        [
            ("img1", "class_a", 0, 0, 100, 100, "test"),
            ("img1", "class_b", 200, 200, 300, 300, "test"),
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "split",
        ],
    )
    preds_df = pd.DataFrame(
        [
            ("img1", "class_a", 0, 0, 100, 100, 0.9),
            ("img1", "class_b", 200, 200, 300, 300, 0.8),
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "confidence",
        ],
    )
    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="test", find_best_confs=True)
    return ev


def test_get_dashboards_creates_missing_directory(tmp_path: Path) -> None:
    """Regression: CI plots are saved into `path`, so it must be created first.

    Previously `os.makedirs` ran only after the plots were written, so a
    not-yet-existing output directory raised FileNotFoundError.
    """
    ev = _simple_evaluation()
    target = tmp_path / "does_not_exist_yet" / "nested"

    ev.get_dashboards(save_to_excel=True, path=str(target), save_confusion_matrix=True)

    files = set(os.listdir(target))
    assert "recall_confidence_intervals.png" in files
    assert "full_dashboard_default.xlsx" in files
    assert "matrix_default.xlsx" in files
