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

Requires Python 3.11+. The core install is `torch`-free. Optional extras add the
`ultralytics` / `torchmetrics` metrics backends (see
[External metrics backends](#external-metrics-backends-single-entry-point)) and
the `clearml` experiment-tracking layer (see
[Experiment tracking (ClearML)](#experiment-tracking-clearml)):

```bash
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[clearml]      @ git+https://github.com/Wasilkas/digital-metrics"
```

`clearml` is `torch`-free; the two backends each pull in `torch`.

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
| `image_path` | `str` | opt | — | Full path to the image file; required **only** by `Evaluation.predict_to_dataframe` (YOLO inference) |
| `image_width` | `int` | opt | — | Image width in pixels; required **only** when `skip_cohen_kappa=False` (Cohen's kappa pixel masks) |
| `image_height` | `int` | opt | — | Image height in pixels; required **only** when `skip_cohen_kappa=False` (Cohen's kappa pixel masks) |

### Input validation

When evaluation runs, inputs are validated and a `ValueError` is raised on:

- missing required columns (per the schema above),
- `NA` values in the predictions `confidence` column,
- prediction `instance_label`s absent from the ground-truth class vocabulary.

(The val/test calibration workflow additionally rejects splits that share an
`image_name`, to prevent calibration leakage.)

---

## Quick start

```python
import pandas as pd
from digital_metrics import Evaluation

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
from digital_metrics import Evaluation, ConfidenceOptimization

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

> If a chosen threshold equals the **minimum** prediction confidence, the cut
> keeps every detection — optimisation had no effect (e.g. predictions matching
> the ground truth so closely that the F1-optimal cut is the floor). A `WARNING`
> is logged per class (or once, for `"global"`) when this happens.

---

## Matching strategies

```python
from digital_metrics import Evaluation, MatchingStrategy

# Default: iou_prior (Ultralytics non-scipy style — IoU-sorted, label-aware)
ev = Evaluation(preds_df, split_df, matching_strategy="iou_prior")

# greedy (YOLO confidence-sorted) or hungarian (globally optimal, geometry-first)
ev = Evaluation(preds_df, split_df, matching_strategy="greedy")
ev = Evaluation(preds_df, split_df, matching_strategy="hungarian")
```

`"iou_prior"` (default) pairs boxes by descending IoU and is label-aware,
mirroring Ultralytics' internal assignment. Use `"greedy"` for confidence-sorted
YOLO-style matching, or `"hungarian"` for annotation-audit workflows where you
want the most plausible pairing between predicted and ground-truth boxes.

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
  is exactly how Ultralytics computes AP. This is the library default.
- **`confidence_optimization="global"`** — YOLO applies a single confidence
  threshold to every class, chosen from the mean-F1 curve.
- **`matching_strategy="iou_prior"`** is the default and mirrors Ultralytics'
  internal IoU-sorted assignment. `"greedy"` is the alternative YOLO-style
  confidence-sorted rule and differs by ~0.006 mAP50 on the fixture data.

### Why other parameters are not a problem

The library is intentionally more flexible than YOLO, and deviating from the
recipe above does **not** make the metrics wrong — it just changes the lens:

- **mAP is invariant to the operating-point knobs.** `mAP50/75/50-95` are always
  computed on the raw, unfiltered predictions over the *entire* precision–recall
  curve, so `preprocess_preds_conf_threshold`, the NMS thresholds and
  `confidence_optimization` have **no effect** on mAP. Those knobs only move the
  single point at which precision/recall/F1 and the confusion matrix are
  reported — every choice yields a valid operating point on the same curve.
- **The AP method barely matters.** `"interp"` (COCO trapezoid, the default) and
  `"continuous"` (exact VOC rectangle area) agree to ≤ 0.001 on the fixture; both
  are standard definitions, so either is defensible.
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

## External metrics backends (single entry point)

To get numbers from an established metrics library — instead of this library's
own `Evaluation` path — use the single entry point `compute_detection_metrics`.
It scores the same GT/prediction DataFrames through one of two optional backends
and returns `dict[str, DetectionMetrics]` (per-class
`precision / recall / f1 / ap50 / ap75 / ap50_95`):

```python
from digital_metrics import compute_detection_metrics

gt_df = split_df[split_df["split"] == "test"]

# YOLO-comparable (Ultralytics' own ap_per_class)
yolo = compute_detection_metrics(gt_df, preds_df, backend="ultralytics")

# General COCO mAP (torchmetrics' MeanAveragePrecision)
coco = compute_detection_metrics(gt_df, preds_df, backend="torchmetrics")

for cls, m in yolo.items():
    print(f"{cls}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
          f"mAP50={m.ap50:.3f} mAP50-95={m.ap50_95:.3f}")
```

- **`backend="ultralytics"`** (default) — YOLO-comparable. Boxes are matched and
  scored by Ultralytics' own `ap_per_class`, so AP equals `model.val()`. P/R/F1
  are read at IoU 0.50 at the single global max-mean-F1 operating point.
- **`backend="torchmetrics"`** — general COCO mAP via torchmetrics'
  `MeanAveragePrecision` (pycocotools). AP is torchmetrics' own
  `map_50 / map_75 / map` per class; P/R/F1 are derived off its IoU-0.50
  precision–recall curve at the per-class max-F1 point (torchmetrics has no
  headline P/R/F1 of its own).

Both backends score only classes that have at least one ground-truth box in the
split. Each is a heavy **optional extra** (each pulls in `torch`), imported
lazily — the core install stays torch-free. Install whichever you need:

```bash
# from a clone
uv sync --extra ultralytics
uv sync --extra torchmetrics

# or directly
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
```

Calling a backend without its extra raises `ImportError` with an install hint; an
unknown `backend` raises `ValueError`. The underlying functions
(`compute_ultralytics_metrics`, `compute_torchmetrics_metrics`) are also public
and callable directly. `YoloMetrics` is kept as a backward-compatible alias of
`DetectionMetrics`.

> These backends are the apples-to-apples comparison path. This library's own
> `Evaluation` P/R/F1 are intentionally custom and are **not** meant to match
> YOLO's console output numerically (see the note above).

On the fixture data the three ways agree on mAP to ~0.002–0.006 but differ on
P/R/F1 by up to ~0.05 — a structural consequence of selecting and reading a single
operating point off the same curve in different ways (per-class vs. one global
threshold; raw vs. COCO-envelope precision). See
[docs/why_prf1_differs.md](docs/why_prf1_differs.md) for the explanation and plots.

---

## `Evaluation` with an external backend

The same two backends are wired into `Evaluation`, so you can choose the metrics
engine and keep the rest of the workflow — dashboards, CI plots, confusion
matrix — unchanged. Pass `backend=` to the constructor, or call a backend
directly:

```python
from digital_metrics import Evaluation

ev = Evaluation(preds_df, split_df, backend="ultralytics")  # or "torchmetrics"
ev(split="test")

ev.detection_metrics   # raw dict[str, DetectionMetrics] from the backend
ev.metrics             # the same numbers adapted to native Metrics
ev.get_dashboards()    # works — built from the backend's results

# Or run a backend without switching the whole Evaluation over:
yolo = ev.compute_metrics_ultralytics(split="test")
coco = ev.compute_metrics_torchmetrics(split="test")
```

- `backend=None` (default) runs the native pipeline. `"ultralytics"` /
  `"torchmetrics"` score the split over the **raw** predictions (the way
  `model.val()` does); `find_best_confs` and the preprocessing thresholds do not
  apply.
- **Calibration** — by default a backend self-selects its operating point on the
  eval split (in-sample). Pass `calibration_split="val"` and the backend instead
  reports P/R/F1 at the F1-optimal confidence found on `val`, reading it off its
  per-class curves; **AP stays over the full curve**, and the chosen threshold(s)
  land on `ev.best_confidences`. `confidence_optimization` selects `"per_class"` vs
  `"global"` thresholds, exactly like the native path. **Both backends support
  this** — `"ultralytics"` reads off `ap_per_class`'s curves, `"torchmetrics"` off
  its `extended_summary` IoU-0.50 precision/score curves. The standalone helpers
  `find_ultralytics_confidence` / `find_torchmetrics_confidence` (with `mode=`) and
  `compute_*_metrics(..., conf_threshold=...)` expose the same mechanism directly.

  ```python
  ev = Evaluation(preds_df, split_df, backend="ultralytics",
                  confidence_optimization="per_class")
  ev(split="test", calibration_split="val")   # calibrate on val, report on test
  ```
- `ev.detection_metrics` holds the untouched backend output; `ev.metrics` holds
  the same precision / recall / f1 / AP **adapted onto native `Metrics`** — TP/FP/FN
  are reconstructed as floats from the per-class GT count so the dashboards and CI
  plots keep working. In this mode `cohen_kappa` is `-1`; the per-class
  `confidence` threshold is `0.0` unless a `calibration_split` set it.
- **Confusion matrix** — the `"ultralytics"` backend fills `ev.cm` /
  `ev.class_labels` using Ultralytics' own confusion-matrix logic (a numpy port of
  `ConfusionMatrix.process_batch`, at its conf 0.25 / IoU 0.45 defaults — the
  matrix `model.val()` plots), transposed to this library's row = GT / column =
  prediction convention. `"torchmetrics"` has no confusion matrix, so `ev.cm` is
  `None` and `get_dashboards` skips that sheet. The standalone
  `compute_ultralytics_confusion_matrix(gt_df, preds_df)` is also public.

---

## From YOLO weights to predictions

If you have an Ultralytics model rather than a predictions table, run inference
straight from the ground-truth DataFrame — `Evaluation.predict_to_dataframe`
closes the eval pipeline at the front, no `data.yaml` needed:

```python
from digital_metrics import Evaluation

# Ground truth must carry an `image_path` column (full path to each image).
# Construct with preds_df=None, then generate predictions from the model:
ev = Evaluation(None, "ground_truth.csv", iou_threshold=0.5)
ev.predict_to_dataframe("best.pt", split="val")   # fills ev.preds_df
ev(split="val")                                    # evaluate as usual
```

- The image source is `split_df["image_path"]`; `image_name` is the last part of
  that path (`Path(image_path).name`), so predictions join back to the ground
  truth automatically. `instance_label` comes from the model's own class names;
  boxes are pixel `xyxy`.
- `split=` chooses which images to run on: a single split (`"val"`), a list of
  splits (`["test", "val"]`), or `None` for every image in `split_df`. (When
  predictions are auto-generated from `weights_path`, `Evaluation` does this for
  you — running only the evaluation split plus any `calibration_split`.)
- The model runs at `conf=0.001`, `iou=0.7` by default (YOLO val settings) so the
  full precision-recall curve is available downstream; raise `conf=` to pre-filter.
- Any extra `model.predict` arguments pass straight through as keyword arguments —
  `ev.predict_to_dataframe("best.pt", split="val", imgsz=1280, half=True, augment=True)`
  — or, in the auto-predict flow, via the constructor's
  `predict_kwargs={"imgsz": 1280, "half": True}`.
- **GPU memory** — inference runs in chunks of `batch` images (default 16), so
  peak VRAM stays bounded (≈ `batch` × per-image cost) instead of growing with the
  image count. If a run OOMs, lower `batch` first, then `imgsz`, and/or set
  `half=True` — e.g. `predict_kwargs={"batch": 4, "imgsz": 1280, "half": True}`.
  (`batch` is a genuine chunk size here; the Ultralytics `batch` predict kwarg is a
  no-op in streaming mode.)
- `predict_to_dataframe` also **returns** the predictions DataFrame, so you can
  save it (`df.to_csv(...)`) or feed it to
  [`compute_detection_metrics`](#external-metrics-backends-single-entry-point).
- `image_name=` selects the `image_name` format (`"name"` filename+ext, the
  default; `"stem"`; or full `"path"`) — match it to your ground-truth `image_name`.

Requires the `ultralytics` extra (imported lazily; the core install stays
torch-free).

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
| `perebrak_ci_lower/upper` | CI on perebrak (1 − precision) |
| `nedobrak_ci_lower/upper` | CI on nedobrak (1 − recall) |

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

Output directories (`path` / the parent of `save_path`) are created
automatically if they don't exist.

### Error audit

```python
# Top-k prediction/GT pairs confused between two classes
audit_df = ev.get_topk_confusions(main_class="car", k=20)

# DataFrames annotated with match type for visualisation
gt_vis, pred_vis = ev.get_dfs_visualization()
```

---

## Experiment tracking (ClearML)

`ClearMLTracker` mirrors a finished `Evaluation` into a
[ClearML](https://clear.ml) task — scalars, artifacts, plots and logs — so runs
are versioned and comparable in the ClearML UI. It is a **standalone layer** that
sits on top of `Evaluation`; the core evaluation code knows nothing about ClearML,
and `clearml` is an optional, `torch`-free extra imported lazily.

```bash
uv pip install "digital-metrics[clearml] @ git+https://github.com/Wasilkas/digital-metrics"
```

```python
from digital_metrics import Evaluation, ClearMLTracker

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)

with ClearMLTracker(project_name="detector", task_name="run-42") as tracker:
    ev(split="test", calibration_split="val")
    tracker.log_evaluation(ev)     # scalars + artifacts + plots + logs
```

`log_evaluation(ev, *, iteration=0, artifacts_dir=None, save_to_excel=True,
save_confusion_matrix=True)` runs `get_dashboards` once and mirrors four things:

- **Scalars** — per-class P/R/F1/mAP as scalar plots, the per-class metrics table,
  and headline means (`mean_*`, nan-aware for AP) as single values.
- **Artifacts** — the analyst/production dashboard DataFrames, `best_confidences`,
  the confusion matrix, and any Excel files `get_dashboards` wrote.
- **Plots** — the four confidence-interval PNGs as images and the confusion matrix
  as a CM plot.
- **Logs** — run logs, via a `loguru` sink attached to the ClearML console.

```python
ClearMLTracker(
    task=None,                 # inject an existing clearml.Task, or let it Task.init one
    *,
    project_name="detector",   # used only when it creates the task
    task_name="run-42",
    output_uri=None,           # where ClearML stores artifacts/models
    attach_logs=True,          # install the loguru → ClearML console sink
    log_level="INFO",
    **task_init_kwargs,        # forwarded to Task.init
)
```

- `clearml` is imported **only** when the tracker creates its own task, so passing
  an existing `task=` needs no extra installed (handy in tests).
- The individual layers are also public: `log_scalars` / `log_artifacts` /
  `log_plots` / `attach_loguru` / `detach_loguru` / `close`. Used as a context
  manager, it closes the task on exit.
- `summarize_metrics(metrics) -> (per_class_df, means)` is the `torch`/ClearML-free
  helper it uses to build the per-class table and nan-aware means; it is public and
  callable on any `dict[str, Metrics]` on its own.

---

## `Evaluation` constructor

```python
Evaluation(
    preds_df: pd.DataFrame | str | None,   # DataFrame, CSV path, or None to predict first
    split_df: pd.DataFrame | str,
    iou_threshold: float = 0.5,
    preprocess: bool = False,       # deduplicate near-identical GT boxes
    skip_cohen_kappa: bool = True,  # kappa is expensive; enable only when needed
    matching_strategy: MatchingStrategy = "iou_prior",  # "iou_prior" | "greedy" | "hungarian"
    preprocess_preds_conf_threshold: float | None = None,
    preprocess_preds_nms_containment_threshold: float | None = None,
    preprocess_preds_nms_iou_threshold: float | None = None,
    ap_method: APMethod = "interp",                              # "interp" | "continuous"
    confidence_optimization: ConfidenceOptimization = "per_class",  # "per_class" | "global"
    weights_path: str | None = None,   # YOLO weights to auto-predict from when preds_df is None
    backend: Backend | None = None,    # None = native; "ultralytics" | "torchmetrics"
    predict_kwargs: dict | None = None,  # extra model.predict(...) kwargs for the weights flow
)
```

The defaults are YOLO-like (`matching_strategy="iou_prior"`, `ap_method="interp"`).

`preds_df` accepts a DataFrame, a CSV path, or `None`. Pass `None` together with
`weights_path` to run the **whole pipeline from weights** — the first call
generates predictions from the model over just the splits it will use (the
evaluation split plus any `calibration_split`), then evaluates:

```python
ev = Evaluation(None, "ground_truth.csv", weights_path="best.pt")
ev(split="val")   # predicts from best.pt, then evaluates
```

If `preds_df` is `None` and no `weights_path` is given, calling the evaluation
raises `ValueError`. You can still predict manually first via
[`predict_to_dataframe`](#from-yolo-weights-to-predictions).

The image scope for each split is derived automatically from the `split`
column in `split_df` — no extra list needs to be passed.

`confidence_optimization` — `"per_class"` (default) tunes one threshold per
class; `"global"` picks a single YOLO-style threshold shared by all classes
(see [Confidence-threshold optimization](#confidence-threshold-optimization)).

`backend` — `None` (default) runs the native pipeline; `"ultralytics"` /
`"torchmetrics"` make `Evaluation` score the split through that external library
instead (see [`Evaluation` with an external backend](#evaluation-with-an-external-backend)).

`predict_kwargs` — extra keyword arguments forwarded to Ultralytics'
`model.predict` when predictions are auto-generated from `weights_path` (e.g.
`{"conf": 0.25, "imgsz": 1280, "half": True, "augment": True}`). Ignored when
`preds_df` is provided. For one-off control, pass the same kwargs straight to
[`predict_to_dataframe`](#from-yolo-weights-to-predictions).

`preprocess_preds_conf_threshold` — drop predictions with `confidence <
threshold` before evaluation.

`preprocess_preds_nms_containment_threshold` — same-class containment
suppression: the lower-confidence box is removed when
`intersection / min(area_a, area_b) >= threshold`.

`preprocess_preds_nms_iou_threshold` — cross-class IoU suppression: the
lower-confidence box is removed when `IoU >= threshold`.

Setting either NMS threshold to `None` disables that suppression type.

### Grouped config objects (optional)

If you'd rather not pass a dozen flat keyword arguments, the constructor also
accepts three optional grouped configs. They are **purely additive** — every flat
kwarg above still works unchanged — and each group, when passed, supplies that
whole group and takes precedence over its corresponding flat kwargs:

```python
from digital_metrics import Evaluation, ScoringConfig, PreprocessConfig, InferenceConfig

ev = Evaluation(
    preds_df,
    split_df,
    scoring=ScoringConfig(iou_threshold=0.5, matching_strategy="greedy"),
    preprocessing=PreprocessConfig(conf_threshold=0.25, nms_iou_threshold=0.5),
    inference=InferenceConfig(weights_path="best.pt", predict_kwargs={"imgsz": 1280}),
)
```

- **`ScoringConfig`** — `iou_threshold`, `matching_strategy`, `ap_method`,
  `confidence_optimization`, `skip_cohen_kappa`.
- **`PreprocessConfig`** — `dedup_gt` (the flat `preprocess`), `conf_threshold`,
  `nms_containment_threshold`, `nms_iou_threshold`.
- **`InferenceConfig`** — `weights_path`, `predict_kwargs`.

Each config's defaults mirror the flat-kwarg defaults, so `Evaluation(preds, split)`
and `Evaluation(preds, split, scoring=ScoringConfig())` behave identically.
`backend` stays a flat top-level argument.

---

## Development

```bash
git clone https://github.com/Wasilkas/digital-metrics
cd digital-metrics
uv venv && uv sync

uv run ruff check . --fix
uv run ruff format .
uv run mypy src/
uv run pytest --cov=src/digital_metrics tests/
```
