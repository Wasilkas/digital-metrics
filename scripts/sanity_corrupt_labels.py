"""Sanity check: corrupt ~half the predictions, expect the F1 optimiser to recover.

Usage:
    uv run python scripts/sanity_corrupt_labels.py

Adapted from a local snippet to run against the repo fixture
(fixtures/ground_truths_all.csv) instead of an absolute machine path.

Construction:
    * preds_df and split_df both start as a verbatim copy of the ground truth.
    * Every prediction gets a random confidence in [0, 1).
    * Every prediction with confidence < 0.5 has its label flipped to "Кромка"
      (a real class in the vocabulary) — i.e. ~50% of predictions are corrupted.
    * GT confidence is set to 1.

Expected sanity outcome (per-class F1 optimisation, find_best_confs=True):
    Because the corruption is gated exactly on confidence < 0.5, the per-class
    optimiser should pick a threshold around 0.5 that drops the corrupted preds.
    Precision then recovers to ~1.0, but recall caps near ~0.5 (the corrupted
    half of each class is gone), so F1 ~ 0.67. "Кромка" additionally absorbs all
    the flipped labels as low-confidence false positives, which the same ~0.5
    threshold suppresses.

    Boxes are identical to GT (IoU == 1.0), so the result is invariant to
    iou_threshold — 0.99, 0.95 and 0.5 must agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from digital_metrics import Evaluation

ROOT = Path(__file__).parent.parent
GT_PATH = ROOT / "fixtures" / "ground_truths_all.csv"
SANITY_DIR = ROOT / "fixtures" / "sanity"

SEED = 0


def f(iou_threshold: float, save_path: str) -> None:
    rng = np.random.default_rng(SEED)

    preds_df = pd.read_csv(GT_PATH, index_col=0)
    split_df = pd.read_csv(GT_PATH, index_col=0)

    preds_df["instance_label"] = preds_df["instance_label"].fillna("")
    split_df["instance_label"] = split_df["instance_label"].fillna("")
    preds_df["confidence"] = rng.random(len(preds_df))
    preds_df.loc[preds_df["confidence"] < 0.5, "instance_label"] = "Кромка"
    split_df["confidence"] = 1

    n_corrupted = int((preds_df["confidence"] < 0.5).sum())
    print(f"\n{'#' * 72}")
    print(
        f"# iou_threshold={iou_threshold}  "
        f"(corrupted {n_corrupted}/{len(preds_df)} preds → 'Кромка')"
    )
    print("#" * 72)

    ev = Evaluation(preds_df, split_df, iou_threshold=iou_threshold)
    ev(split="test", find_best_confs=True)

    # Per-class metrics
    for cls, m in ev.metrics.items():
        print(
            f"{cls}: P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1_score:.3f}  mAP50={m.ap50:.3f}"
        )

    # Confidence thresholds chosen to maximise per-class F1
    print("best_confidences:", {k: round(v, 3) for k, v in ev.best_confidences.items()})

    # Confusion matrix
    assert ev.cm is not None
    print("cm shape:", ev.cm.shape)

    # ── Compact summary for interpretation ───────────────────────────────────
    mean_p = float(np.mean([m.precision for m in ev.metrics.values()]))
    mean_r = float(np.mean([m.recall for m in ev.metrics.values()]))
    mean_f1 = float(np.mean([m.f1_score for m in ev.metrics.values()]))
    mean_ap50 = float(np.nanmean([m.ap50 for m in ev.metrics.values()]))
    print(f"MEAN  P={mean_p:.3f}  R={mean_r:.3f}  F1={mean_f1:.3f}  mAP50={mean_ap50:.3f}")

    # Export dashboards (Excel + CI plots + confusion matrix) under fixtures/sanity.
    out_dir = SANITY_DIR / save_path
    ev.get_dashboards(
        save_to_excel=True,
        path=str(out_dir),
        save_confusion_matrix=True,
    )
    print(f"dashboards written to: {out_dir}")


if __name__ == "__main__":
    f(0.99, "iou-0.99")
    f(0.95, "iou-0.95")
    f(0.5, "iou-0.50")
