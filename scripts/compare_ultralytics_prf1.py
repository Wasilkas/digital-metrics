"""Compare our library's per-class P / R / F1 against Ultralytics' *own* metrics.

We have no model weights or source images, so we cannot run ``model.val()``.
Instead both sides are produced from the *same* GT / prediction boxes:

* Ultralytics side: ``metrics.compute_ultralytics_metrics`` — the library's
  optional YOLO-exact path. It feeds the boxes through Ultralytics' own matching
  (a faithful per-threshold re-match) and metric code
  (``ultralytics.utils.metrics.ap_per_class``), so the P/R/F1 are *theirs*.
* Our side: ``metrics.Evaluation`` at the equivalent operating point —
  ``matching_strategy="iou_prior"`` (the per-class equivalent of Ultralytics'
  ``match_predictions``), ``confidence_optimization="global"`` (one YOLO-style
  threshold shared by all classes), with in-sample threshold selection on the
  same ``test`` split (``find_best_confs=True``), mirroring how Ultralytics picks
  its operating point on the evaluation set itself.

Ultralytics' headline P/R/F1 are read at IoU 0.50 off a 1000-point interpolated
P-R curve at the single global confidence that maximises the smoothed mean
per-class F1. The matching is equivalent, so residual per-class differences come
only from how that operating point is read — interpolated/smoothed curve vs our
realizable confidence tie-group boundaries (CLAUDE.md: "Do not try to make P/R/F1
numerically match YOLO's console output").

Run:
    uv run --with ultralytics python scripts/compare_ultralytics_prf1.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from digital_metrics import Evaluation, compute_ultralytics_metrics

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"

SPLIT = "test"
IOU_THRESHOLD = 0.5


def main() -> None:
    logger.info("Loading data...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)

    test_gt = gt_df[gt_df["split"] == SPLIT]
    test_images = set(test_gt["image_name"].unique())
    test_preds = preds_df[preds_df["image_name"].isin(test_images)]
    logger.info(
        f"test: {test_gt['image_name'].nunique()} images, "
        f"{len(test_gt)} GT boxes, {len(test_preds)} predictions"
    )

    # Shared class vocabulary so both sides agree on which classes exist.
    classes = sorted(gt_df["instance_label"].unique())

    # ── Ultralytics side: P/R/F1 from the library's YOLO-exact path ───────────
    logger.info("Computing P/R/F1 with Ultralytics (compute_ultralytics_metrics)...")
    ult = compute_ultralytics_metrics(test_gt, test_preds, classes=classes)
    ult_prf1 = {c: (m.precision, m.recall, m.f1) for c, m in ult.items()}

    # ── Our side: iou_prior + global YOLO-style threshold, in-sample on test ──
    logger.info("Computing P/R/F1 with our library (iou_prior, global conf)...")
    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=IOU_THRESHOLD,
        matching_strategy="iou_prior",
        confidence_optimization="global",
    )
    ev(split=SPLIT, find_best_confs=True)  # in-sample global max-mean-F1 threshold
    ours = {c: (m.precision, m.recall, m.f1_score) for c, m in ev.metrics.items()}

    # ── Per-class comparison, sorted by F1 disagreement ──────────────────────
    shared = sorted(set(ours) & set(ult_prf1))
    rows = []
    for c in shared:
        op, orr, of1 = ours[c]
        up, ur, uf1 = ult_prf1[c]
        rows.append(
            {
                "class": c,
                "ours_p": op,
                "ours_r": orr,
                "ours_f1": of1,
                "ult_p": up,
                "ult_r": ur,
                "ult_f1": uf1,
                "d_f1": of1 - uf1,
            }
        )
    rows.sort(key=lambda r: abs(r["d_f1"]), reverse=True)

    print("\n" + "=" * 92)
    print(f"{'CLASS':<22} {'ours P/R/F1':>26} {'ultralytics P/R/F1':>26} {'ΔF1':>8}")
    print("=" * 92)
    for r in rows:
        ours_s = f"{r['ours_p']:.3f}/{r['ours_r']:.3f}/{r['ours_f1']:.3f}"
        ult_s = f"{r['ult_p']:.3f}/{r['ult_r']:.3f}/{r['ult_f1']:.3f}"
        print(f"{r['class']:<22} {ours_s:>26} {ult_s:>26} {r['d_f1']:>+8.4f}")
    print("=" * 92)

    d_f1 = np.array([r["d_f1"] for r in rows])
    print(f"\nclasses compared (present as GT): {len(rows)}")
    print(f"per-class F1  max|diff|={np.abs(d_f1).max():.5f}  mean|diff|={np.abs(d_f1).mean():.5f}")

    labels = ("P", "R", "F1")
    ours_mean = {k: np.mean([ours[c][i] for c in shared]) for i, k in enumerate(labels)}
    ult_mean = {k: np.mean([ult_prf1[c][i] for c in shared]) for i, k in enumerate(labels)}
    print(
        f"\nmean over classes (IoU 0.50, global max-F1 operating point):"
        f"\n  ours        P={ours_mean['P']:.4f}  R={ours_mean['R']:.4f}  F1={ours_mean['F1']:.4f}"
        f"\n  ultralytics P={ult_mean['P']:.4f}  R={ult_mean['R']:.4f}  F1={ult_mean['F1']:.4f}\n"
    )


if __name__ == "__main__":
    main()
