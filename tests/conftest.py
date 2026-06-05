import pandas as pd
import pytest


@pytest.fixture
def tiny_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two images, three classes, designed so TP/FP/FN counts are exact.

    Image 1 GTs:
      1. class_a  (0,   0, 100, 100)
      2. class_b  (200, 200, 300, 300)
      3. class_c  (400, 400, 500, 500)

    Image 1 Preds (sorted by conf):
      - class_a (0,   0, 100, 100) conf=0.90  → TP  (IoU=1.0 with GT#1)
      - class_b (200, 200, 300, 300) conf=0.80 → TP  (IoU=1.0 with GT#2)
      - class_a (150, 150, 250, 250) conf=0.30 → FP  (IoU≈0.14 with GT#2, < 0.5)

    GT#3 (class_c) has no pred → FN

    Image 2 GTs:
      1. class_a  (0,   0, 100, 100)
      2. class_a  (200, 200, 300, 300)
      3. class_b  (400, 400, 500, 500)
      4. class_c  (600, 600, 700, 700)

    Image 2 Preds:
      - class_a (0,   0, 100, 100) conf=0.95 → TP  (IoU=1.0 with GT#1)
      - class_a (200, 200, 300, 300) conf=0.85 → TP (IoU=1.0 with GT#2)
      - class_c (600, 600, 700, 700) conf=0.60 → TP  (IoU=1.0 with GT#4)
      - class_a (250, 250, 350, 350) conf=0.20 → FP  (IoU≈0.14 with GT#2 already consumed)

    GT#3 (class_b) has no pred → FN

    Expected counts at IoU=0.5, no confidence filtering:
      class_a: TP=3, FP=2, FN=0
      class_b: TP=1, FP=0, FN=1
      class_c: TP=1, FP=0, FN=1
    """
    gt_rows = [
        # image_name, instance_label, x_tl, y_tl, x_br, y_br, split
        ("img1", "class_a", 0, 0, 100, 100, "test"),
        ("img1", "class_b", 200, 200, 300, 300, "test"),
        ("img1", "class_c", 400, 400, 500, 500, "test"),
        ("img2", "class_a", 0, 0, 100, 100, "test"),
        ("img2", "class_a", 200, 200, 300, 300, "test"),
        ("img2", "class_b", 400, 400, 500, 500, "test"),
        ("img2", "class_c", 600, 600, 700, 700, "test"),
    ]
    gt_df = pd.DataFrame(
        gt_rows,
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

    pred_rows = [
        # image_name, instance_label, x_tl, y_tl, x_br, y_br, confidence
        ("img1", "class_a", 0, 0, 100, 100, 0.90),
        ("img1", "class_b", 200, 200, 300, 300, 0.80),
        ("img1", "class_a", 150, 150, 250, 250, 0.30),
        ("img2", "class_a", 0, 0, 100, 100, 0.95),
        ("img2", "class_a", 200, 200, 300, 300, 0.85),
        ("img2", "class_c", 600, 600, 700, 700, 0.60),
        ("img2", "class_a", 250, 250, 350, 350, 0.20),
    ]
    preds_df = pd.DataFrame(
        pred_rows,
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

    return gt_df, preds_df


@pytest.fixture
def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dataset with explicit 'val' and 'test' splits for calibration testing.

    Val images (img1, img2):
      GT: class_a (img1), class_b (img1), class_a (img2)
      Preds: all perfect matches at high confidence

    Test images (img3, img4):
      GT: class_a (img3), class_b (img3), class_a (img4)
      Preds: all perfect matches at lower confidence

    Using calibration_split="val" should produce the same thresholds as
    running find_best_confs=True directly on the val split.
    """
    gt_rows = [
        ("img1", "class_a", 0, 0, 100, 100, "val"),
        ("img1", "class_b", 200, 200, 300, 300, "val"),
        ("img2", "class_a", 0, 0, 100, 100, "val"),
        ("img3", "class_a", 0, 0, 100, 100, "test"),
        ("img3", "class_b", 200, 200, 300, 300, "test"),
        ("img4", "class_a", 0, 0, 100, 100, "test"),
    ]
    gt_df = pd.DataFrame(
        gt_rows,
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
    pred_rows = [
        # Val images — high confidence
        ("img1", "class_a", 0, 0, 100, 100, 0.90),
        ("img1", "class_b", 200, 200, 300, 300, 0.85),
        ("img2", "class_a", 0, 0, 100, 100, 0.88),
        # Test images — lower confidence
        ("img3", "class_a", 0, 0, 100, 100, 0.70),
        ("img3", "class_b", 200, 200, 300, 300, 0.65),
        ("img4", "class_a", 0, 0, 100, 100, 0.60),
    ]
    preds_df = pd.DataFrame(
        pred_rows,
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
    return gt_df, preds_df
