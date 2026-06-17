"""Local evaluation script.

Usage:
    uv run python scripts/eval.py

Loads fixtures/ground_truths_all.csv and fixtures/predicts_all.csv,
calibrates confidence thresholds on the val split, evaluates on the test
split, then writes results to fixtures/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from metrics import Evaluation

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"


def main() -> None:
    logger.info("Loading data...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)

    logger.info(f"GT rows: {len(gt_df)}  |  Preds rows: {len(preds_df)}")
    logger.info(f"GT splits: {sorted(gt_df['split'].unique().tolist())}")

    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=0.3,
        skip_cohen_kappa=True,
        # Pinned to the documented baseline config (greedy / continuous AP);
        # the library defaults are now YOLO-like (iou_prior / interp).
        matching_strategy="greedy",
        ap_method="continuous",
        preprocess_preds_conf_threshold=0.1,
        preprocess_preds_nms_containment_threshold=0.9,
        preprocess_preds_nms_iou_threshold=0.6,
    )

    logger.info("Running evaluation (calibrate on val, evaluate on test)...")
    ev(split="test", calibration_split="val")

    # ── Per-class metrics DataFrame ──────────────────────────────────────────
    rows = []
    for cls, m in ev.metrics.items():
        rows.append(
            {
                "class": cls,
                "tp": m.tp,
                "fp": m.fp,
                "fn": m.fn,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1_score": round(m.f1_score, 4),
                "perebrak": round(m.perebrak, 4),
                "nedobrak": round(m.nedobrak, 4),
                "ap50": round(m.ap50, 4) if not np.isnan(m.ap50) else float("nan"),
                "ap75": round(m.ap75, 4) if not np.isnan(m.ap75) else float("nan"),
                "ap50_95": round(m.ap50_95, 4) if not np.isnan(m.ap50_95) else float("nan"),
                "confidence": round(m.confidence, 4),
                "precision_ci_lower": round(m.precision_ci_lower, 4),
                "precision_ci_upper": round(m.precision_ci_upper, 4),
                "recall_ci_lower": round(m.recall_ci_lower, 4),
                "recall_ci_upper": round(m.recall_ci_upper, 4),
            }
        )

    metrics_df = pd.DataFrame(rows).set_index("class")
    out_metrics = FIXTURES / "eval_metrics.csv"
    metrics_df.to_csv(out_metrics)
    logger.info(f"Saved per-class metrics → {out_metrics}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    assert ev.cm is not None
    cm_df = pd.DataFrame(ev.cm, index=ev.class_labels, columns=ev.class_labels)
    out_cm = FIXTURES / "eval_confusion_matrix.csv"
    cm_df.to_csv(out_cm)
    logger.info(f"Saved confusion matrix → {out_cm}")

    # ── Best confidence thresholds ───────────────────────────────────────────
    out_confs = FIXTURES / "eval_best_confidences.json"
    out_confs.write_text(
        json.dumps(
            {cls: round(conf, 4) for cls, conf in ev.best_confidences.items()},
            ensure_ascii=False,
            indent=2,
        )
    )
    logger.info(f"Saved best confidences → {out_confs}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'CLASS':<30} {'P':>7} {'R':>7} {'F1':>7} {'mAP50':>7} {'CONF':>7}")
    print("=" * 72)
    for cls, m in sorted(ev.metrics.items()):
        ap50_str = f"{m.ap50:.4f}" if not np.isnan(m.ap50) else "  nan "
        print(
            f"{cls:<30} {m.precision:>7.4f} {m.recall:>7.4f} "
            f"{m.f1_score:>7.4f} {ap50_str:>7} {m.confidence:>7.4f}"
        )
    print("=" * 72)

    valid_ap50 = [m.ap50 for m in ev.metrics.values() if not np.isnan(m.ap50)]
    valid_ap50_95 = [m.ap50_95 for m in ev.metrics.values() if not np.isnan(m.ap50_95)]
    mean_p = np.mean([m.precision for m in ev.metrics.values()])
    mean_r = np.mean([m.recall for m in ev.metrics.values()])
    mean_f1 = np.mean([m.f1_score for m in ev.metrics.values()])
    print(
        f"\nmean (all classes):  P={mean_p:.4f}  R={mean_r:.4f}  F1={mean_f1:.4f}"
        f"  mAP50={np.mean(valid_ap50):.4f}  mAP50-95={np.mean(valid_ap50_95):.4f}"
    )
    print(f"classes evaluated: {len(ev.metrics)}\n")


if __name__ == "__main__":
    main()
