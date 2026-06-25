"""Illustrate why mAP agrees across the three metric paths but P/R/F1 don't.

Writes two figures to ``docs/``:

* ``prf1_means.png`` — mean P/R/F1 vs mAP50/mAP50-95 per way. The mAP bars line
  up; the P/R/F1 bars don't.
* ``prf1_pr_curve.png`` — one class's precision-recall curve: the raw (realizable)
  curve, the COCO precision *envelope*, F1 iso-lines, and the operating points the
  different paths read. mAP is the area under the envelope (the same for all
  three); P/R/F1 are a single *point* on the curve, chosen/read differently.

Run (both extras needed for the external bars; the P-R curve needs only core):
    uv run --with ultralytics --with 'torchmetrics[detection]' \
        python scripts/plot_prf1_vs_map.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from digital_metrics import Evaluation, compute_detection_metrics
from digital_metrics.matching import match_boxes
from digital_metrics.scoring import find_best_global_confidence
from digital_metrics.types import PredictMatch

plt.switch_backend("Agg")

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "fixtures"
DOCS = ROOT / "docs"
GT_PATH = FIXTURES / "ground_truths_all.csv"
PREDS_PATH = FIXTURES / "predicts_all.csv"

SPLIT = "test"
IOU = 0.5
BAR_LABELS = ("P", "R", "F1", "mAP50", "mAP50-95")
COLORS = {"ours": "#4C72B0", "ultralytics": "#DD8452", "torchmetrics": "#55A868"}


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gt_df = pd.read_csv(GT_PATH, index_col=0)
    preds_df = pd.read_csv(PREDS_PATH, index_col=0)
    test_gt = gt_df[gt_df["split"] == SPLIT]
    test_images = set(test_gt["image_name"].unique())
    test_preds = preds_df[preds_df["image_name"].isin(test_images)]
    return gt_df, test_gt, test_preds


def run_evaluation(gt_df: pd.DataFrame) -> Evaluation:
    ev = Evaluation(
        preds_df=pd.read_csv(PREDS_PATH, index_col=0),
        split_df=gt_df,
        iou_threshold=IOU,
        matching_strategy="iou_prior",
        ap_method="interp",
        confidence_optimization="per_class",  # library default: one F1-optimal thr per class
    )
    ev(split=SPLIT, find_best_confs=True)
    return ev


def collect_means(
    ev: Evaluation, gt_df: pd.DataFrame, test_gt: pd.DataFrame, test_preds: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Mean (P, R, F1, mAP50, mAP50-95) over shared classes, per way."""
    classes = sorted(gt_df["instance_label"].unique())
    ways: dict[str, dict[str, tuple[float, ...]]] = {
        "ours": {
            c: (m.precision, m.recall, m.f1_score, m.ap50, m.ap50_95)
            for c, m in ev.metrics.items()
            if not np.isnan(m.ap50)
        }
    }
    for backend in ("ultralytics", "torchmetrics"):
        try:
            res = compute_detection_metrics(test_gt, test_preds, backend=backend, classes=classes)
            ways[backend] = {
                c: (m.precision, m.recall, m.f1, m.ap50, m.ap50_95) for c, m in res.items()
            }
        except ImportError as exc:
            logger.warning(f"[{backend}] skipped — {exc}")

    shared = sorted(set.intersection(*(set(w) for w in ways.values())))
    logger.info(f"means over {len(shared)} shared classes across {len(ways)} way(s)")
    return {name: np.mean([w[c] for c in shared], axis=0) for name, w in ways.items()}


def plot_means(means: dict[str, np.ndarray]) -> Path:
    ways = list(means)
    x = np.arange(len(BAR_LABELS), dtype=float)
    bar_w = 0.8 / len(ways)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    for i, name in enumerate(ways):
        offset = (i - (len(ways) - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means[name], bar_w, label=name, color=COLORS.get(name))
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    # Visually split the operating-point-dependent group from the curve-area group.
    ax.axvline(2.5, color="grey", ls=":", lw=1)
    ax.text(
        1.0,
        1.02,
        "single operating point\n(differs by path)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    ax.text(
        3.5,
        1.02,
        "area under the curve\n(path-invariant)",
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(BAR_LABELS)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("mean over classes (test split)")
    ax.set_title("Same matching → mAP agrees; the P/R/F1 operating point does not")
    ax.legend(loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = DOCS / "prf1_means.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _pr_curve(records: list[PredictMatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-sorted recall, precision and confidence for one class."""
    preds = sorted(
        (m for m in records if m.type in ("TP", "FP")),
        key=lambda m: m.confidence,
        reverse=True,
    )
    n_gt = sum(1 for m in records if m.type in ("TP", "FN"))
    tp = np.array([1.0 if m.type == "TP" else 0.0 for m in preds])
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1.0 - tp)
    recall = ctp / max(n_gt, 1)
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    conf = np.array([m.confidence for m in preds])
    return recall, precision, conf


def _f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    return 2.0 * p * r / np.maximum(p + r, 1e-9)


def _idx_at(conf: np.ndarray, thr: float) -> int:
    """Last realizable curve index when keeping predictions with conf >= thr."""
    return int((conf >= thr).sum()) - 1


def plot_pr_curve(
    ev: Evaluation, test_gt: pd.DataFrame, test_preds: pd.DataFrame, shared: list[str]
) -> Path:
    matches = match_boxes(
        test_gt,
        test_preds,
        IOU,
        strategy="iou_prior",
        split_image_names=test_gt["image_name"].unique().tolist(),
    )
    # The dominant P/R/F1 lever is operating-point *selection*: per-class max-F1
    # (where ours & torchmetrics sit) vs one global threshold (where ultralytics
    # sits). Pick a well-populated class where those two points separate in recall.
    global_thr = find_best_global_confidence(matches, shared)
    min_pred = 60
    best_cls, best_gap = shared[0], -1.0
    for c in shared:
        recs = matches.get(c, [])
        if sum(1 for m in recs if m.type in ("TP", "FP")) < min_pred:
            continue
        if ev.metrics[c].ap50 < 0.5:  # keep the curve well-shaped (not a bottom-hugger)
            continue
        rec, _, conf = _pr_curve(recs)
        kp, kg = _idx_at(conf, ev.best_confidences[c]), _idx_at(conf, global_thr)
        if kp < 0 or kg < 0:
            continue
        gap = abs(rec[kp] - rec[kg])
        if gap > best_gap:
            best_cls, best_gap = c, gap
    cls = best_cls

    recall, precision, conf = _pr_curve(matches[cls])
    envelope = np.maximum.accumulate(precision[::-1])[::-1]  # COCO/VOC precision envelope
    kp = _idx_at(conf, ev.best_confidences[cls])  # ours — per-class max-F1 (raw)
    kg = _idx_at(conf, global_thr)  # ultralytics / ours-global — single shared threshold
    j = int(np.argmax(_f1(envelope, recall)))  # torchmetrics — per-class max-F1 (envelope)

    fig, ax = plt.subplots(figsize=(8.5, 6.8), dpi=120)

    # F1 iso-lines: p = f*r / (2r - f) for 2r > f.
    rr = np.linspace(0.001, 1.0, 400)
    for f in (0.4, 0.6, 0.8):
        pp = np.where(2 * rr > f, f * rr / np.maximum(2 * rr - f, 1e-9), np.nan)
        pp[(pp < 0) | (pp > 1)] = np.nan
        ax.plot(rr, pp, color="lightgrey", lw=1, ls="--", zorder=1)
        idx = np.nanargmin(np.abs(rr - 0.97))
        if not np.isnan(pp[idx]):
            ax.text(0.975, pp[idx], f"F1={f}", color="grey", fontsize=8, va="center")

    ax.step(
        recall, precision, where="post", color="#4C72B0", lw=1.6, label="raw P-R curve (realizable)"
    )
    ax.plot(
        recall,
        envelope,
        color="#C44E52",
        lw=1.4,
        ls="-",
        label="COCO precision envelope (area = AP)",
    )
    ax.fill_between(recall, envelope, step=None, alpha=0.06, color="#C44E52")

    def _lab(name: str, p: float, r: float) -> str:
        return f"{name}\n  P={p:.3f} R={r:.3f} F1={_f1(np.array([p]), np.array([r]))[0]:.3f}"

    # Per-class max-F1 — where ours (raw) and torchmetrics (envelope) both sit;
    # they nearly coincide, so the raw↔envelope readout is only a hair apart.
    ax.scatter(
        [recall[kp]],
        [precision[kp]],
        s=110,
        color="#4C72B0",
        zorder=6,
        edgecolor="white",
        label=_lab("ours — per-class max-F1 (raw)", precision[kp], recall[kp]),
    )
    ax.scatter(
        [recall[j]],
        [envelope[j]],
        s=80,
        color="#C44E52",
        zorder=6,
        marker="D",
        edgecolor="white",
        label=_lab("torchmetrics — per-class max-F1 (envelope)", envelope[j], recall[j]),
    )
    # Single global threshold — where ultralytics (and ours-global) put this class.
    ax.scatter(
        [recall[kg]],
        [precision[kg]],
        s=110,
        color="#DD8452",
        zorder=6,
        marker="s",
        edgecolor="white",
        label=_lab(
            f"single global thr {global_thr:.3f} (ultralytics / ours-global)",
            precision[kg],
            recall[kg],
        ),
    )
    ax.annotate(
        "",
        xy=(recall[kp], precision[kp]),
        xytext=(recall[kg], precision[kg]),
        arrowprops={"arrowstyle": "->", "color": "grey", "lw": 1.2},
    )

    ap50 = ev.metrics[cls].ap50
    ax.set_xlim(0, 1.04)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title(
        f"Class «{cls}»  (n_pred={len(conf)}, AP50={ap50:.3f})\n"
        "same curve (same mAP); the P/R/F1 point moves with threshold selection"
    )
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = DOCS / "prf1_pr_curve.png"
    fig.savefig(out)
    plt.close(fig)
    logger.info(
        f"PR-curve class «{cls}»: per-class(R={recall[kp]:.3f},P={precision[kp]:.3f}) "
        f"vs global thr {global_thr:.3f}(R={recall[kg]:.3f},P={precision[kg]:.3f})"
    )
    return out


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    logger.info("Loading data and running Evaluation...")
    gt_df, test_gt, test_preds = load()
    ev = run_evaluation(gt_df)
    shared = sorted(c for c, m in ev.metrics.items() if not np.isnan(m.ap50))

    means = collect_means(ev, gt_df, test_gt, test_preds)
    p1 = plot_means(means)
    p2 = plot_pr_curve(ev, test_gt, test_preds, shared)
    logger.info(f"wrote {p1}")
    logger.info(f"wrote {p2}")


if __name__ == "__main__":
    main()
