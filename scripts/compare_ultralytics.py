"""Compare our library's mAP against Ultralytics' own metric implementation.

We have no model weights or source images, so we cannot run ``model.val()``.
Instead we feed the *same* GT / prediction boxes through Ultralytics' own
matching + AP code and compare against ``metrics.Evaluation`` on the identical
test split:

    * ``ultralytics.utils.metrics.box_iou`` — IoU matrix per image.
    * ``BaseValidator.match_predictions`` (non-scipy path, copied verbatim
      below) — assigns predictions to GT across 10 IoU thresholds.
    * ``ultralytics.utils.metrics.ap_per_class`` — the actual COCO/VOC AP code
      Ultralytics uses to produce mAP50 and mAP50-95.

To make the comparison exact we mirror what ``compute_map`` does internally
(see src/metrics/ap.py): evaluate on the ``test`` GT split and keep only the
predictions whose ``image_name`` appears in that split. Our library is then run
with the documented YOLO recipe (``matching_strategy="iou_prior"``,
``ap_method="interp"``), which is the per-class equivalent of Ultralytics'
``match_predictions``.

Run:
    uv run --with ultralytics python scripts/compare_ultralytics.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from ultralytics.utils.metrics import ap_per_class, box_iou

from metrics import Evaluation

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"

SPLIT = "test"
BOX = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
IOUV = torch.linspace(0.5, 0.95, 10)  # 10 IoU thresholds, exactly as YOLO val


def match_predictions(
    pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor
) -> np.ndarray:
    """Copy of ``BaseValidator.match_predictions`` (non-scipy path, v8.4.70).

    ``iou`` is the (n_gt, n_pred) IoU matrix. Returns an (n_pred, 10) bool array
    marking, for each IoU threshold, whether each prediction is a correct match.
    """
    correct = np.zeros((pred_classes.shape[0], IOUV.shape[0])).astype(bool)
    correct_class = true_classes[:, None] == pred_classes  # (n_gt, n_pred)
    iou = iou * correct_class  # zero out the wrong classes
    iou = iou.cpu().numpy()
    for i, threshold in enumerate(IOUV.tolist()):
        matches = np.nonzero(iou >= threshold)
        matches = np.array(matches).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def ultralytics_ap(
    gt_df: pd.DataFrame, preds_df: pd.DataFrame, class_to_idx: dict[str, int]
) -> dict[int, np.ndarray]:
    """Run Ultralytics' own matching + ap_per_class over the box set.

    Returns ``{class_idx: ap_row}`` where each ``ap_row`` is the length-10 AP
    vector (AP at IoU 0.50 … 0.95) Ultralytics computes for that class.
    """
    tp_all: list[np.ndarray] = []
    conf_all: list[np.ndarray] = []
    pred_cls_all: list[np.ndarray] = []
    target_cls_all: list[np.ndarray] = []

    gt_groups = {name: g for name, g in gt_df.groupby("image_name")}
    pred_groups = {name: g for name, g in preds_df.groupby("image_name")}

    for image_name, g in gt_groups.items():
        gt_boxes = torch.tensor(g[BOX].to_numpy(np.float32))
        gt_cls = g["instance_label"].map(class_to_idx).to_numpy(np.int64)
        # Every GT box is a target (drives recall / FN), even with no preds.
        target_cls_all.append(gt_cls)

        p = pred_groups.get(image_name)
        if p is None or len(p) == 0:
            continue

        pred_boxes = torch.tensor(p[BOX].to_numpy(np.float32))
        pred_cls = p["instance_label"].map(class_to_idx).to_numpy(np.int64)
        conf = p["confidence"].to_numpy(np.float32)

        iou = box_iou(gt_boxes, pred_boxes)  # (n_gt, n_pred)
        correct = match_predictions(torch.tensor(pred_cls), torch.tensor(gt_cls), iou)
        tp_all.append(correct)
        conf_all.append(conf)
        pred_cls_all.append(pred_cls)

    tp = np.concatenate(tp_all, 0)
    conf = np.concatenate(conf_all, 0)
    pred_cls = np.concatenate(pred_cls_all, 0)
    target_cls = np.concatenate(target_cls_all, 0)

    results = ap_per_class(tp, conf, pred_cls, target_cls)
    ap = results[5]  # (n_classes_with_gt, 10)
    unique_classes = results[6]  # int class indices, sorted
    return {int(c): ap[i] for i, c in enumerate(unique_classes)}


def main() -> None:
    logger.info("Loading data...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)

    # ── Mirror compute_map's view of the data ────────────────────────────────
    test_gt = gt_df[gt_df["split"] == SPLIT]
    test_images = set(test_gt["image_name"].unique())
    test_preds = preds_df[preds_df["image_name"].isin(test_images)]
    logger.info(
        f"test: {test_gt['image_name'].nunique()} images, "
        f"{len(test_gt)} GT boxes, {len(test_preds)} predictions"
    )

    # Stable class vocabulary shared by both sides.
    classes = sorted(gt_df["instance_label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}

    # ── Ultralytics side ─────────────────────────────────────────────────────
    logger.info("Computing AP with Ultralytics' own code...")
    ult_ap = ultralytics_ap(test_gt, test_preds, class_to_idx)
    ult_ap50 = {idx_to_class[i]: row[0] for i, row in ult_ap.items()}
    ult_ap5095 = {idx_to_class[i]: float(row.mean()) for i, row in ult_ap.items()}

    # ── Our library (documented YOLO recipe) ─────────────────────────────────
    logger.info("Computing AP with our library (iou_prior / interp)...")
    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=0.5,
        matching_strategy="iou_prior",
        ap_method="interp",
    )
    ev(split=SPLIT, find_best_confs=False)
    ours_ap50 = {c: m.ap50 for c, m in ev.metrics.items() if not np.isnan(m.ap50)}
    ours_ap5095 = {c: m.ap50_95 for c, m in ev.metrics.items() if not np.isnan(m.ap50_95)}

    # ── Per-class comparison (AP50), sorted by abs diff ──────────────────────
    shared = sorted(set(ours_ap50) & set(ult_ap50))
    rows = [
        {
            "class": c,
            "ours_ap50": ours_ap50[c],
            "ult_ap50": ult_ap50[c],
            "diff": ours_ap50[c] - ult_ap50[c],
        }
        for c in shared
    ]
    rows.sort(key=lambda r: abs(r["diff"]), reverse=True)

    print("\n" + "=" * 64)
    print(f"{'CLASS':<28} {'OURS':>8} {'ULTRA':>8} {'DIFF':>9}")
    print("=" * 64)
    for r in rows:
        print(f"{r['class']:<28} {r['ours_ap50']:>8.4f} {r['ult_ap50']:>8.4f} {r['diff']:>+9.5f}")
    print("=" * 64)

    diffs = np.array([r["diff"] for r in rows])
    print(f"\nclasses compared: {len(rows)}")
    print(
        f"per-class AP50  max|diff|={np.abs(diffs).max():.5f}  "
        f"mean|diff|={np.abs(diffs).mean():.5f}"
    )

    print(
        f"\nmAP50      ours={np.mean(list(ours_ap50.values())):.4f}  "
        f"ultralytics={np.mean(list(ult_ap50.values())):.4f}"
    )
    print(
        f"mAP50-95   ours={np.mean(list(ours_ap5095.values())):.4f}  "
        f"ultralytics={np.mean(list(ult_ap5095.values())):.4f}\n"
    )


if __name__ == "__main__":
    main()
