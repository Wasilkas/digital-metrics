# digital-metrics

Object detection evaluation library. Computes per-class detection metrics
(precision, recall, F1, mAP50 / mAP75 / mAP50-95, Cohen's kappa, Wilson
confidence intervals) from pandas DataFrames of ground-truth and predictions.
Outputs `Metrics` objects, a confusion matrix, Excel dashboards, and CI plots.

> 🇷🇺 Русская версия: [README.ru.md](README.ru.md)

---

## Installation

```bash
uv pip install git+https://github.com/Wasilkas/digital-metrics
# or
pip install git+https://github.com/Wasilkas/digital-metrics
```

Requires Python 3.11+.

---

## Input schema

Both DataFrames share the same column names:

| Column | Type | GT | Preds | Description |
|---|---|:---:|:---:|---|
| `image_name` | `str` | ✓ | ✓ | Unique image identifier |
| `instance_label` | `str` | ✓ | ✓ | Class name |
| `bbox_x_tl` | `float` | ✓ | ✓ | Bounding-box top-left x |
| `bbox_y_tl` | `float` | ✓ | ✓ | Bounding-box top-left y |
| `bbox_x_br` | `float` | ✓ | ✓ | Bounding-box bottom-right x |
| `bbox_y_br` | `float` | ✓ | ✓ | Bounding-box bottom-right y |
| `split` | `str` | ✓ | — | `"train"` / `"val"` / `"test"` |
| `confidence` | `float` | — | ✓ | Detection score in `[0, 1]` |

---

## Quick start

```python
import pandas as pd
from metrics import Evaluation

preds_df = pd.read_csv("predictions.csv", index_col=0)
split_df = pd.read_csv("ground_truth.csv", index_col=0)

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", find_best_confs=True)

# Per-class metrics
for cls, m in ev.metrics.items():
    print(f"{cls}: P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1_score:.3f}  mAP50={m.ap50:.3f}")

# Confidence thresholds chosen to maximise per-class F1
print(ev.best_confidences)

# Confusion matrix
print(ev.cm)          # ndarray (n_classes+1, n_classes+1)
print(ev.class_labels)
```

---

## Val-calibrated thresholds (recommended)

Find optimal confidence thresholds on the validation split, then evaluate
on test — no in-sample optimism:

```python
ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", calibration_split="val")
```

---

## Confidence-threshold optimization

When `find_best_confs=True` (or a `calibration_split` is given), confidence
thresholds are tuned automatically. Two modes are available via
`confidence_optimization=`:

```python
from metrics import Evaluation, ConfidenceOptimization

# Default: one threshold per class, each maximising that class's F1
ev = Evaluation(preds_df, split_df, confidence_optimization="per_class")

# YOLO-style: a single threshold shared by all classes
ev = Evaluation(preds_df, split_df, confidence_optimization="global")
ev(split="test", calibration_split="val")
```

- **`"per_class"`** (default) — `ev.best_confidences` holds a *different*
  threshold per class, each chosen to maximise that class's F1. Best for
  squeezing per-class quality out of a model.
- **`"global"`** — mirrors Ultralytics YOLO, which applies **one** confidence
  threshold to every class. The threshold that maximises the **mean per-class
  F1** is selected and applied uniformly, so every entry in
  `ev.best_confidences` is identical. Use this when reporting numbers that must
  be comparable to YOLO's, or when production inference runs a single `conf`
  value.

Both modes honour the val-calibration workflow: thresholds are found on the
calibration split and applied to the evaluation split.

---

## Matching strategies

```python
from metrics import Evaluation, MatchingStrategy

# Default: greedy (YOLO-style, confidence-sorted)
ev = Evaluation(preds_df, split_df, matching_strategy="greedy")

# Optional: Hungarian (globally optimal geometry-first assignment)
ev = Evaluation(preds_df, split_df, matching_strategy="hungarian")
```

Use `"hungarian"` for annotation-audit workflows where you want the most
plausible pairing between predicted and ground-truth boxes.

---

## Predictions preprocessing

Apply confidence filtering and/or custom NMS before evaluation by passing
thresholds to the constructor:

```python
ev = Evaluation(
    preds_df,
    split_df,
    # Drop low-confidence predictions
    preprocess_preds_conf_threshold=0.25,
    # Suppress same-class box that is largely inside another (containment >= 0.8)
    preprocess_preds_nms_containment_threshold=0.8,
    # Suppress lower-confidence box when two different-class boxes overlap (IoU >= 0.5)
    preprocess_preds_nms_iou_threshold=0.5,
)
```

Each threshold is independent — set only the ones you need. Setting a threshold
to `None` (the default) disables that suppression type.

---

## Reproducing YOLO (Ultralytics) metrics

To get numbers that line up as closely as possible with an Ultralytics
`model.val()` run, mirror the model's inference config and pick the YOLO-style
options:

```python
ev = Evaluation(
    preds_df,
    split_df,
    iou_threshold=0.5,                       # report P/R/F1 at IoU 0.50 (mAP50 operating point)
    preprocess_preds_conf_threshold=0.001,   # same minimum confidence as the model config
    preprocess_preds_nms_iou_threshold=0.7,  # same NMS IoU as the model config
    ap_method="interp",                      # 101-point trapezoid integration (COCO / Ultralytics)
    confidence_optimization="global",        # one confidence threshold for all classes
)
ev(split="test", calibration_split="val")
```

- **`preprocess_preds_conf_threshold` / `preprocess_preds_nms_iou_threshold`** —
  set these to the `conf` and `iou` values from your YOLO config (val defaults
  are `conf=0.001`, `iou=0.7`) so predictions enter matching at the same
  operating point the model used. If `preds_df` was *exported* from the model
  (NMS already applied), you can leave the NMS threshold at `None` — re-applying
  it is just a safety net for cross-class duplicates.
- **`ap_method="interp"`** — the 101-point trapezoid integral (`np.trapezoid`)
  is exactly how Ultralytics computes AP.
- **`confidence_optimization="global"`** — YOLO applies a single confidence
  threshold to every class, chosen from the mean-F1 curve.
- For the strictest match on the mAP path, you can also pass
  `matching_strategy="iou_prior"`, which mirrors Ultralytics' internal
  IoU-sorted assignment; the default `"greedy"` is also YOLO-style and differs
  by ~0.006 mAP50 on the fixture data.

### Why other parameters are not a problem

The library is intentionally more flexible than YOLO, and deviating from the
recipe above does **not** make the metrics wrong — it just changes the lens:

- **mAP is invariant to the operating-point knobs.** `mAP50/75/50-95` are always
  computed on the raw, unfiltered predictions over the *entire* precision–recall
  curve, so `preprocess_preds_conf_threshold`, the NMS thresholds and
  `confidence_optimization` have **no effect** on mAP. Those knobs only move the
  single point at which precision/recall/F1 and the confusion matrix are
  reported — every choice yields a valid operating point on the same curve.
- **The AP method barely matters.** `"continuous"` (exact VOC rectangle area)
  and `"interp"` (COCO trapezoid) agree to ≤ 0.001 on the fixture; both are
  standard definitions, so the default `"continuous"` is equally defensible.
- **Per-class confidence can only match or beat global.** Global thresholding is
  the constrained special case of per-class (one shared value vs. the best value
  per class), so `"per_class"` gives an equal-or-higher mean F1 and a finer
  operating point — without touching mAP.
- **All matching strategies are valid assignment rules.** greedy / iou_prior /
  hungarian differ only marginally on mAP; pick by intent (greedy/iou_prior for
  YOLO parity, hungarian for annotation audits).

In short: use the recipe when you need figures directly comparable to an
Ultralytics run; otherwise the defaults (or per-class confidence) give a richer,
often more favourable view while keeping mAP exactly as comparable.

---

## Outputs

### `ev.metrics` — `dict[str, Metrics]`

Each `Metrics` object exposes:

| Field | Description |
|---|---|
| `tp`, `fp`, `fn` | True positives / false positives / false negatives |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | 2 · P · R / (P + R) |
| `perebrak` | 1 − precision (false-positive rate) |
| `nedobrak` | 1 − recall (miss rate) |
| `ap50` | AP at IoU = 0.50 (`nan` when class absent from split) |
| `ap75` | AP at IoU = 0.75 (`nan` when class absent from split) |
| `ap50_95` | mAP averaged over IoU 0.50 … 0.95 (`nan` when class absent from split) |
| `cohen_kappa` | Cohen's kappa via pixel-mask method |
| `confidence` | Best confidence threshold for this class |
| `precision_ci_lower/upper` | Wilson 95 % CI on precision |
| `recall_ci_lower/upper` | Wilson 95 % CI on recall |

### Dashboards and plots

```python
# Excel dashboards + optional confusion-matrix image
summary_df, detail_df = ev.get_dashboards(
    save_to_excel=True,
    path="/path/to/output/",
    save_confusion_matrix=True,
)

# Confidence-interval bar chart
fig, ax = ev.plot_confidence_intervals(
    metric="precision",         # or "recall"
    confidence_level=0.95,
    save_path="/path/to/ci_plot.png",
)
```

### Error audit

```python
# Top-k prediction/GT pairs confused between two classes
audit_df = ev.get_topk_confusions(main_class="car", k=20)

# DataFrames annotated with match type for visualisation
gt_vis, pred_vis = ev.get_dfs_visualization()
```

---

## `Evaluation` constructor

```python
Evaluation(
    preds_df: pd.DataFrame,
    split_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    preprocess: bool = False,       # deduplicate near-identical GT boxes
    skip_cohen_kappa: bool = True,  # kappa is expensive; enable only when needed
    matching_strategy: MatchingStrategy = "greedy",
    preprocess_preds_conf_threshold: float | None = None,
    preprocess_preds_nms_containment_threshold: float | None = None,
    preprocess_preds_nms_iou_threshold: float | None = None,
    ap_method: APMethod = "continuous",                          # "continuous" | "interp"
    confidence_optimization: ConfidenceOptimization = "per_class",  # "per_class" | "global"
)
```

The image scope for each split is derived automatically from the `split`
column in `split_df` — no extra list needs to be passed.

`confidence_optimization` — `"per_class"` (default) tunes one threshold per
class; `"global"` picks a single YOLO-style threshold shared by all classes
(see [Confidence-threshold optimization](#confidence-threshold-optimization)).

`preprocess_preds_conf_threshold` — drop predictions with `confidence <
threshold` before evaluation.

`preprocess_preds_nms_containment_threshold` — same-class containment
suppression: the lower-confidence box is removed when
`intersection / min(area_a, area_b) >= threshold`.

`preprocess_preds_nms_iou_threshold` — cross-class IoU suppression: the
lower-confidence box is removed when `IoU >= threshold`.

Setting either NMS threshold to `None` disables that suppression type.

---

## Development

```bash
git clone https://github.com/Wasilkas/digital-metrics
cd digital-metrics
uv venv && uv sync

uv run ruff check . --fix
uv run ruff format .
uv run mypy src/
uv run pytest --cov=src/metrics tests/
```
