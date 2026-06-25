"""Compare all three metric paths on the same fixture boxes.

Three "ways" to score the identical GT / prediction boxes on the test split:

1. ``Evaluation`` — this library's own pipeline at the documented YOLO recipe
   (``matching_strategy="iou_prior"``, ``ap_method="interp"``,
   ``confidence_optimization="global"``, in-sample threshold on test).
2. ``ultralytics`` — ``compute_detection_metrics(backend="ultralytics")``,
   Ultralytics' own ``ap_per_class``.
3. ``torchmetrics`` — ``compute_detection_metrics(backend="torchmetrics")``,
   torchmetrics' ``MeanAveragePrecision`` (COCO mAP).

All three report per-class precision / recall / f1 (at IoU 0.50) and
ap50 / ap75 / ap50-95. P/R/F1 are read at each path's own max-F1 operating point,
so small differences are expected (interpolated curves vs. realizable thresholds);
mAP is the apples-to-apples number.

The two external backends are optional extras (both pull in torch); a backend
whose dependency is missing is skipped with a note, so the script runs with
whatever is installed.

Run (with both extras):
    uv run --with ultralytics --with 'torchmetrics[detection]' \
        python scripts/compare_backends.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from digital_metrics import Backend, Evaluation, compute_detection_metrics

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"

SPLIT = "test"
IOU_THRESHOLD = 0.5

METRIC_LABELS = ("P", "R", "F1", "mAP50", "mAP75", "mAP50-95")
# Short column headers so the per-class tables stay readable in a terminal.
DISPLAY = {"ours": "ours", "ultralytics": "ultra", "torchmetrics": "torch"}


@dataclass
class Scores:
    """Per-class metrics shared by all three ways."""

    precision: float
    recall: float
    f1: float
    ap50: float
    ap75: float
    ap50_95: float

    def as_tuple(self) -> tuple[float, ...]:
        return (self.precision, self.recall, self.f1, self.ap50, self.ap75, self.ap50_95)


def compute_ours(preds_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Scores]:
    """Score with this library's own Evaluation pipeline (YOLO recipe)."""
    ev = Evaluation(
        preds_df=preds_df,
        split_df=gt_df,
        iou_threshold=IOU_THRESHOLD,
        matching_strategy="iou_prior",
        ap_method="interp",
        confidence_optimization="global",
    )
    ev(split=SPLIT, find_best_confs=True)  # in-sample global max-mean-F1 threshold
    out: dict[str, Scores] = {}
    for c, m in ev.metrics.items():
        if np.isnan(m.ap50):
            continue  # class absent from the test split
        out[c] = Scores(m.precision, m.recall, m.f1_score, m.ap50, m.ap75, m.ap50_95)
    return out


def compute_external(
    backend: Backend, test_gt: pd.DataFrame, test_preds: pd.DataFrame, classes: list[str]
) -> dict[str, Scores]:
    """Score with one external backend via the single entry point."""
    res = compute_detection_metrics(test_gt, test_preds, backend=backend, classes=classes)
    return {
        c: Scores(m.precision, m.recall, m.f1, m.ap50, m.ap75, m.ap50_95) for c, m in res.items()
    }


def print_summary(ways: dict[str, dict[str, Scores]], shared: list[str]) -> None:
    """Mean over shared classes, one row per way, all six metrics."""
    width = 14 + 10 * len(METRIC_LABELS)
    print("\n" + "=" * width)
    print(f"{'WAY':<14}" + "".join(f"{lbl:>10}" for lbl in METRIC_LABELS))
    print("=" * width)
    for name, scores in ways.items():
        means = np.mean([scores[c].as_tuple() for c in shared], axis=0)
        print(f"{name:<14}" + "".join(f"{v:>10.4f}" for v in means))
    print("=" * width)


def print_per_class(
    ways: dict[str, dict[str, Scores]], shared: list[str], attr: str, title: str
) -> None:
    """One metric, per class, one column per way, sorted by spread across ways."""
    names = list(ways)
    width = 20 + 10 * len(names) + 9
    print(f"\n{title} (per class, sorted by spread across ways)")
    print("-" * width)
    print(f"{'CLASS':<20}" + "".join(f"{DISPLAY.get(n, n):>10}" for n in names) + f"{'spread':>9}")
    print("-" * width)
    rows = [(c, [getattr(ways[n][c], attr) for n in names]) for c in shared]
    rows.sort(key=lambda r: max(r[1]) - min(r[1]), reverse=True)
    for c, vals in rows:
        spread = max(vals) - min(vals)
        print(f"{c:<20}" + "".join(f"{v:>10.4f}" for v in vals) + f"{spread:>9.4f}")
    print("-" * width)


def main() -> None:
    logger.info("Loading data...")
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)

    test_gt = gt_df[gt_df["split"] == SPLIT]
    test_images = set(test_gt["image_name"].unique())
    test_preds = preds_df[preds_df["image_name"].isin(test_images)]
    classes = sorted(gt_df["instance_label"].unique())
    logger.info(
        f"test: {test_gt['image_name'].nunique()} images, "
        f"{len(test_gt)} GT boxes, {len(test_preds)} predictions"
    )

    ways: dict[str, dict[str, Scores]] = {}

    logger.info("[ours] Evaluation (iou_prior / interp / global conf)...")
    ways["ours"] = compute_ours(preds_df, gt_df)

    for backend in ("ultralytics", "torchmetrics"):
        logger.info(f"[{backend}] compute_detection_metrics...")
        try:
            ways[backend] = compute_external(backend, test_gt, test_preds, classes)
        except ImportError as exc:
            logger.warning(f"[{backend}] skipped — {exc}")

    if len(ways) < 2:
        logger.error(
            "No external backend installed — nothing to compare against 'ours'. "
            "Install one: uv pip install 'ultralytics' or 'torchmetrics[detection]'."
        )

    shared = sorted(set.intersection(*(set(w) for w in ways.values())))
    logger.info(f"classes shared across {len(ways)} way(s): {len(shared)}")

    print_summary(ways, shared)
    print_per_class(ways, shared, "ap50_95", "mAP50-95")
    print_per_class(ways, shared, "f1", "F1 @ IoU 0.50")


if __name__ == "__main__":
    main()
