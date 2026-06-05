# digital-metrics

Object detection evaluation library. Computes per-class detection metrics
(precision, recall, F1, mAP50 / mAP75 / mAP50-95, Cohen's kappa, Wilson
confidence intervals) from pandas DataFrames of ground-truth and predictions.
Outputs `Metrics` objects, a confusion matrix, Excel dashboards, and CI plots.

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
| `ap50` | AP at IoU = 0.50 |
| `ap75` | AP at IoU = 0.75 |
| `ap50_95` | mAP averaged over IoU 0.50 … 0.95 |
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
)
```

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
