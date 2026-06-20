"""Identity sanity check: predictions == ground truth.

Usage:
    uv run python scripts/eval_identity.py

Builds a predictions DataFrame that is a verbatim copy of the ground truth
(same boxes, same labels, same images) with random confidence scores attached,
then runs the evaluation. If the matching/metric plumbing is correct, every
class must score precision = recall = f1 = mAP50 = mAP50-95 = 1.0.

Notes:
    * No preprocessing (confidence filter / NMS) is applied — any of those would
      drop or merge prediction rows and break the verbatim-copy premise,
      manufacturing false negatives.
    * The evaluation is in-sample on split="all": confidence optimisation then
      picks the all-keeping threshold, so recall reaches 1.0. A val->test
      calibration with *random* confidences would drop some test preds below the
      val-derived threshold (a sampling artifact, not a real miss).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from metrics import Evaluation

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"

SEED = 0


def main() -> None:
    logger.info("Loading ground truth...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    logger.info(f"GT rows: {len(gt_df)}  |  classes: {gt_df['instance_label'].nunique()}")

    # Predictions = verbatim copy of GT + random confidences.
    rng = np.random.default_rng(SEED)
    preds_df = gt_df.copy()
    preds_df["confidence"] = rng.uniform(0.0, 1.0, size=len(preds_df))

    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=0.5,
        skip_cohen_kappa=True,
        # No preprocessing: keep preds a verbatim copy of GT.
    )

    logger.info("Running identity evaluation (in-sample, split='all')...")
    ev(split="all", find_best_confs=True)

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'CLASS':<30} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7} {'mAP5095':>8}")
    print("=" * 80)
    imperfect: list[str] = []
    for cls, m in sorted(ev.metrics.items()):
        ap50 = m.ap50
        ap5095 = m.ap50_95
        print(
            f"{cls:<30} {m.precision:>7.4f} {m.recall:>7.4f} {m.f1_score:>7.4f} "
            f"{ap50:>7.4f} {ap5095:>8.4f}"
        )
        if not (
            np.isclose(m.precision, 1.0)
            and np.isclose(m.recall, 1.0)
            and np.isclose(m.f1_score, 1.0)
            and np.isclose(ap50, 1.0)
            and np.isclose(ap5095, 1.0)
        ):
            imperfect.append(cls)
    print("=" * 80)

    mean_p = np.mean([m.precision for m in ev.metrics.values()])
    mean_r = np.mean([m.recall for m in ev.metrics.values()])
    mean_f1 = np.mean([m.f1_score for m in ev.metrics.values()])
    mean_ap50 = np.nanmean([m.ap50 for m in ev.metrics.values()])
    mean_ap5095 = np.nanmean([m.ap50_95 for m in ev.metrics.values()])
    print(
        f"\nmean (all classes):  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}"
        f"  mAP50={mean_ap50:.4f}  mAP50-95={mean_ap5095:.4f}"
    )
    print(f"classes evaluated: {len(ev.metrics)}")

    total_tp = sum(m.tp for m in ev.metrics.values())
    total_fp = sum(m.fp for m in ev.metrics.values())
    total_fn = sum(m.fn for m in ev.metrics.values())
    print(
        f"totals:  TP={total_tp:.0f}  FP={total_fp:.0f}  FN={total_fn:.0f}  (GT rows={len(gt_df)})"
    )

    if imperfect:
        print(f"\n[FAIL] {len(imperfect)} class(es) did not score a perfect 1.0:")
        for cls in imperfect:
            print(f"  - {cls}")
    else:
        print("\n[OK] identity holds: every class scored P=R=F1=mAP50=mAP50-95=1.0")


if __name__ == "__main__":
    main()
