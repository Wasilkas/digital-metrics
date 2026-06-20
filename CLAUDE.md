# CLAUDE.md

## Project Overview

Object detection evaluation library. Computes detection metrics (precision, recall, mAP50/75/50-95,
F1, Cohen's kappa, confidence intervals) from pandas DataFrames of ground-truth and predictions.
Output: per-class `Metrics` objects, confusion matrix, Excel dashboards, CI plots.

---

## Environment

- **Python**: 3.11+
- **Package manager**: `uv` with lockfile (`uv.lock`)
- **Virtual environment**: `uv venv` (`.venv/`)

```bash
uv venv
uv sync          # install from lockfile
uv add <pkg>     # add dependency (updates pyproject.toml + uv.lock)
uv run pytest    # run tests inside venv

uv sync --extra ultralytics    # optional YOLO-comparable metrics backend (heavy: torch)
uv sync --extra torchmetrics   # optional COCO-mAP metrics backend (heavy: torch)
```

Never use `pip` directly. Always use `uv`.

---

## Code Style

### Linting & type-checking

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint + autofix
uv run ruff format .         # format
uv run mypy src/             # type-check
```

Ruff config is in `pyproject.toml`. mypy runs in strict mode.

### General principles

- **Simple over clever**: readable code is preferred over micro-optimisations.
  Optimise only where profiling shows a bottleneck.
- **Explicit types**: all public functions must have full type annotations.
- **Small, focused functions**: each function does one thing.
- **No dead code**: remove commented-out blocks, unused imports, TODO stubs.
- **Logging**: use `loguru`. No bare `print()` calls.
- **No mixed languages in logs/comments**: use English throughout.

---

## Project Structure

Modules are grouped into subpackages by role. Each subpackage's ``__init__.py``
re-exports its public names, so internal and external code imports from the
subpackage (e.g. ``from .scoring import compute_map``), and the top-level
``metrics/__init__.py`` keeps the public API flat (``from metrics import ...``).
``types.py``, ``ci.py`` and ``validation.py`` stay at the top level as the shared
foundation (``types`` depends on ``ci``), alongside the ``evaluation`` orchestrator,
the ``calibration`` collaborator it delegates threshold selection to, and the
``engines`` it dispatches scoring to (``NativeEngine`` / ``BackendEngine``).

```
src/
  metrics/
    __init__.py       # public API (re-exports from the subpackages below)
    types.py          # Pydantic models: PredictMatch, Metrics, DetectionMetrics
    ci.py             # Wilson confidence interval (foundation; types depends on it)
    validation.py     # validate_dataframes: shared GT/preds column + label checks
    config.py         # ScoringConfig/PreprocessConfig/InferenceConfig (optional grouped Evaluation args)
    evaluation.py     # Evaluation orchestrator: data/IO + dispatch to a scoring engine
    calibration.py    # ConfidenceCalibrator: threshold selection (delegated by Evaluation)
    engines/          # pluggable scoring engines selected by Evaluation's backend
      __init__.py     # re-exports: ScoringEngine, ScoringInputs, EvaluationResult, Native/BackendEngine
      base.py         # ScoringEngine protocol + ScoringInputs/EvaluationResult dataclasses
      native.py       # NativeEngine: match → calibrate → slice → metrics/mAP/kappa/CM
      backend.py      # BackendEngine: external library scoring + adapt onto native Metrics
    matching/         # box matching: geometry → assignment → records
      __init__.py     # re-exports: compute_iou_matrix, match_boxes, MatchingStrategy, assign_*
      iou.py          # IoU matrix computation
      assignment.py   # pure box-assignment kernels (greedy/iou_prior/hungarian) on IoU matrices
      matching.py     # box matching → PredictMatch records; wraps assignment kernels
    scoring/          # metric computations from matches/boxes
      __init__.py     # re-exports: compute_map/compute_ap/APMethod, compute_kappa,
                      #   find_best_*/ConfidenceOptimization, get_confusion_matrix/get_confusions
      ap.py           # AP / mAP computation; APMethod + MatchingStrategy options; reuses assignment kernels
      confidence.py   # best-confidence search: per-class + global (YOLO-style) thresholds
      kappa.py        # Cohen's kappa (pixel-mask method)
      confusion.py    # confusion matrix helpers (native, match-record based)
    preprocess/       # predictions preprocessing
      __init__.py     # re-exports: filter_by_confidence, apply_nms, PredictionPreprocessor
      nms.py          # confidence filter + custom NMS
      preprocessor.py # PredictionPreprocessor: conf filter + NMS (delegated by Evaluation)
    reporting/        # output artifacts
      __init__.py     # re-exports: get_dashboards, plot_confidence_intervals
      dashboard.py    # Excel export + CI plots
    backends/         # optional, torch-backed metric backends (lazy torch import)
      __init__.py     # re-exports: compute_detection_metrics/Backend, compute_ultralytics_*, compute_torchmetrics_metrics, YoloMetrics
      external.py     # single entry point: compute_detection_metrics(backend=...)
      ultralytics_metrics.py  # optional YOLO-comparable metrics (ap_per_class) + confusion matrix
      torchmetrics_metrics.py # optional COCO mAP via torchmetrics MeanAveragePrecision
    inference/        # optional YOLO inference (lazy torch import)
      __init__.py     # re-exports: predict_on_images, ImageNameMode
      yolo_predict.py # YOLO inference helper (Evaluation.predict_to_dataframe)
tests/
  conftest.py
  test_iou.py
  test_matching.py
  test_ap.py
  test_ci.py
  test_evaluation.py
  test_nms.py
  test_preprocessor.py          # PredictionPreprocessor: enabled flag, conf/NMS, no-op, no-mutate
  test_confidence.py
  test_confidence_calibrator.py # ConfidenceCalibrator: leak check, dispatch, warning, parity
  test_engines.py               # NativeEngine/BackendEngine: run result, resolve split, selection
  test_config.py                # grouped config objects: defaults, grouped==flat, precedence
  test_external_metrics.py     # dispatcher; ValueError path runs without extras
  test_ultralytics_metrics.py  # optional; skipped unless `ultralytics` is installed
  test_torchmetrics_metrics.py # optional; skipped unless `torchmetrics` is installed
  test_yolo_predict.py         # predict_to_dataframe: torch-free helpers, image_path/ImportError guards
scripts/
  eval.py             # local evaluation script (see "Local Evaluation" section)
fixtures/
  ground_truths_all.csv     # GT data (val + test splits, 20 557 rows, 49 classes)
  predicts_all.csv          # model predictions (21 280 rows)
  eval_metrics.csv          # per-class metrics output (generated by eval.py)
  eval_confusion_matrix.csv # confusion matrix output
  eval_best_confidences.json # val-calibrated thresholds per class
pyproject.toml
uv.lock
CLAUDE.md
```

---

## DataFrame Schema

Both DataFrames share these columns:

| Column | Type | Description |
|---|---|---|
| `image_name` | str | Unique image identifier |
| `instance_label` | str | Class name |
| `bbox_x_tl` | float | Bounding box top-left x |
| `bbox_y_tl` | float | Bounding box top-left y |
| `bbox_x_br` | float | Bounding box bottom-right x |
| `bbox_y_br` | float | Bounding box bottom-right y |
| `split` | str | `train` / `val` / `test` (GT only) |
| `confidence` | float | Detection score (predictions only) |
| `image_path` | str | Full path to the image file (GT only; required only by `Evaluation.predict_to_dataframe`) |
| `image_width` | int | Image width in pixels (GT only; required only when `skip_cohen_kappa=False`) |
| `image_height` | int | Image height in pixels (GT only; required only when `skip_cohen_kappa=False`) |

---

## Metric Definitions

### Standard metrics (YOLO-compatible)

- **IoU** — `intersection_area / union_area`
- **TP** — IoU ≥ threshold, correct label, GT not yet matched (one match per GT)
- **FP** — no GT matched (IoU below threshold, or all candidates already taken)
- **FN** — GT box with no matching prediction
- **Precision** — `TP / (TP + FP)`
- **Recall** — `TP / (TP + FN)`
- **F1** — `2 * P * R / (P + R)`
- **perebrak** — `1 - precision` (domain term; false-positive rate)
- **nedobrak** — `1 - recall` (domain term; miss rate)
- **CI** — Wilson interval on precision / recall

### mAP

mAP is computed **independently** from the per-threshold matching used for P/R/F1.
It uses `_raw_preds_df` (unpreprocessed predictions) and runs its own inner matching
loop for each IoU threshold, mirroring the Ultralytics two-path design:

- Sort all predictions by confidence descending (globally per class).
- For each IoU threshold in `[0.50, 0.55, …, 0.95]` (10 values):
  - Match using the configured `strategy` (greedy, iou_prior, or hungarian).
  - Accumulate cumulative TP/FP → precision-recall curve.
- **AP** — area under P-R curve, method configurable (see AP Methods below).
- **mAP50** — AP at IoU = 0.50
- **mAP75** — AP at IoU = 0.75
- **mAP50-95** — mean of AP over all 10 thresholds

Classes with **no GT instances in the evaluated split** receive `float("nan")`
for `ap50`, `ap75`, and `ap50_95` — not `0.0`. This allows `nanmean` to correctly
exclude absent classes from averages.

**Preprocessing split**: confidence filtering and NMS are applied to `self.preds_df`
(used for P/R/F1/CM), but `compute_map` always receives `self._raw_preds_df`
(unfiltered original predictions), matching the Ultralytics design.

### AP Methods (`APMethod`)

Two AP integration methods are available via `ap_method=` on `Evaluation`. The
`Evaluation` constructor defaults to `"interp"` (YOLO-like); the lower-level
`compute_map` still defaults to `"continuous"`.

- **`"interp"`** (Evaluation default) — 101-point COCO interpolation,
  Ultralytics-compatible sentinels (`mpre[0] = 1.0`, `mrec[-1] = recall[-1] + 1e-4`),
  integrates with `np.trapezoid` over 101 equally-spaced recall points. Returns
  0.0 on empty recall.
- **`"continuous"`** — VOC 2010+ rectangle-area integration. Prepends `(0, 0)`
  and appends `(1, 0)` sentinels, right-to-left precision envelope, sums rectangle
  areas at recall change points.

On the fixture dataset the two methods differ by ≤ 0.001 on mean mAP50.

### What is NOT identical to YOLO

Precision, recall, F1, and the confusion matrix are computed from `match_boxes`
which runs once at a single IoU threshold with optional confidence filtering and
label-aware TP classification. YOLO does not expose per-threshold P/R/F1 in the
same way. These metrics are intentionally custom. Do not try to make them
numerically match YOLO's console output.

### External metrics backends (optional, single entry point)

`backends/external.py` exposes one dispatcher for library-backed metrics:

```python
Backend = Literal["ultralytics", "torchmetrics"]

def compute_detection_metrics(
    gt_df, preds_df, *, backend="ultralytics", classes=None, split_image_names=None,
) -> dict[str, DetectionMetrics]:
    ...
```

Both backends return `dict[str, DetectionMetrics]` (per-class
`precision/recall/f1/ap50/ap75/ap50_95`), scored only on classes with at least one
GT box. Each backend is a heavy **optional extra** (both pull in `torch`),
imported lazily; the core install stays torch-free. The underlying functions
(`compute_ultralytics_metrics`, `compute_torchmetrics_metrics`) are also public
and callable directly.

`Evaluation` is the unified entry point over these backends: pass
`backend="ultralytics"`/`"torchmetrics"` (or call
`evaluation.compute_metrics_ultralytics(split)` /
`compute_metrics_torchmetrics(split)` directly). In `backend` mode the call scores
the split over `_raw_preds_df`, stores the raw result on `detection_metrics`, and
adapts it onto native `Metrics` (reconstructing float TP/FP/FN from the per-class
GT count) so `get_dashboards` / `plot_confidence_intervals` work unchanged. By
default a backend self-selects its operating point on the eval split;
`find_best_confs` does not apply. Passing `calibration_split` enables proper
"calibrate on val, report on test" for the `"ultralytics"` backend: the F1-optimal
confidence is found on the calibration split (`find_ultralytics_confidence`, per
`confidence_optimization` mode) and `compute_ultralytics_metrics(..., conf_threshold=)`
reads the eval split's P/R/F1 at that confidence off the ap_per_class curves — AP
stays over the full curve, and the chosen threshold(s) land on `best_confidences`.
`"torchmetrics"` ignores `calibration_split` with a warning (not supported yet).
The `"ultralytics"` backend also fills `cm` / `class_labels` via
`compute_ultralytics_confusion_matrix` (Ultralytics' `ConfusionMatrix.process_batch`,
ported; conf 0.25 / IoU 0.45, transposed to the sklearn row=GT/col=pred convention);
`"torchmetrics"` has no CM (`cm` is `None`).

- **`"ultralytics"`** (`backends/ultralytics_metrics.py`, `pip install
  digital-metrics[ultralytics]`) — YOLO-comparable. Boxes go through a faithful
  per-threshold re-match (`_match_predictions`, a numpy port of
  `BaseValidator.match_predictions`) and Ultralytics' `box_iou`, then Ultralytics'
  own `ap_per_class`, so the numbers equal `model.val()`. P/R/F1 are read off the
  smoothed 1000-point P-R curve at the single global max-mean-F1 threshold (IoU
  0.50). On the fixture test split our `Evaluation` P/R/F1 (`iou_prior` + `global`
  confidence) track this to mean |ΔF1| ≈ 0.007. See
  `scripts/compare_ultralytics_prf1.py`.
- **`"torchmetrics"`** (`backends/torchmetrics_metrics.py`, `pip install
  digital-metrics[torchmetrics]`) — general COCO mAP from torchmetrics'
  `MeanAveragePrecision` (pycocotools backend). AP is torchmetrics' own
  `map_50`/`map_75`/`map` per class. torchmetrics is AP-native with no headline
  P/R/F1, so we derive them the YOLO way: off its `extended_summary` IoU-0.50
  101-point P-R curve at the per-class max-F1 recall point.

`YoloMetrics` is kept as a backward-compatible alias of `DetectionMetrics` (same
fields); `compute_ultralytics_metrics` still returns those objects.

**Comparison scripts** (all read the fixture `test` split; external backends
skipped when their extra is absent):
- `scripts/compare_backends.py` — all three ways side by side (`Evaluation` vs
  `ultralytics` vs `torchmetrics`): mean summary + per-class P/R/F1 and mAP.
- `scripts/compare_ultralytics.py` — `Evaluation` mAP vs the `ultralytics` backend.
- `scripts/compare_ultralytics_prf1.py` — `Evaluation` P/R/F1 vs the `ultralytics`
  backend.
- `scripts/plot_prf1_vs_map.py` — renders the figures in `docs/why_prf1_differs.md`
  (means bars + a per-class P-R curve) explaining why mAP agrees across the three
  ways but P/R/F1 don't (operating-point + raw-vs-COCO-envelope readout).

---

## Box Matching — Three Strategies

`matching/matching.py` exposes three strategies behind a common interface:

```python
MatchingStrategy = Literal["greedy", "hungarian", "iou_prior"]

def match_boxes(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    strategy: MatchingStrategy = "greedy",
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    ...
```

`split_image_names` is an internal parameter populated by `Evaluation._call`
from `gt_df["image_name"].unique()`. External callers can pass `None`.

Note: `match_boxes` (and `compute_map`) default to `"greedy"`, but the
`Evaluation` constructor defaults to the YOLO-like `"iou_prior"`.

### Greedy (YOLO-style)

1. Sort predictions by confidence descending.
2. For each prediction, find the highest-IoU unmatched GT.
3. If IoU ≥ threshold → GT consumed; labels match → TP, else → FP with GT label.
4. Otherwise → FP with `gt_label="background"`.
5. Any unmatched GT → FN.

Used for P/R/F1/CM and for the mAP inner loop when selected as the strategy.

### IoU-Prior (Evaluation default, Ultralytics non-scipy style)

1. Find all pred-GT pairs where IoU ≥ threshold **and** labels match.
2. Sort by IoU descending.
3. Assign greedily: each pred and each GT matched at most once; highest-IoU pair wins.
4. Unmatched preds → FP (cross-class closest GT recorded for CM if IoU ≥ threshold
   and labels differ; otherwise `"background"`).
5. Unmatched GTs → FN.

Key difference from greedy: **confidence plays no role in pairing**. A lower-confidence
pred with better IoU wins the GT. On the fixture dataset vs greedy: recall drops ~0.08
(GTs with label-mismatch preds are no longer silently consumed), mAP50 drops ~0.006.

**In `compute_map`**: since the loop is already per-class, label matching is
automatic — iou_prior simply sorts all pairs with IoU ≥ threshold by IoU and
assigns greedily, keeping the result in confidence-sorted index order for the P-R curve.

### Hungarian (globally optimal)

Uses `scipy.optimize.linear_sum_assignment` on the negative IoU matrix.
Geometry-first, confidence-independent. More expensive: O(N³) per image.

**Supported in `compute_map`** — runs `linear_sum_assignment` per image then maps
results back to confidence-sorted index order for the P-R curve.

---

## Tests

- Use `pytest` + `pytest-cov`.
- Place fixtures in `tests/conftest.py`.
- Cover:
  - IoU edge cases (perfect overlap, no overlap, partial)
  - Greedy matching: hand-computed TP/FP/FN on `tiny_dataset`
  - IoU-prior matching: same `tiny_dataset` counts; discriminating fixture that
    verifies lower-conf/higher-IoU pred wins the GT (opposite of greedy)
  - Hungarian matching: same `tiny_dataset`, verify total TP ≥ greedy TP
  - AP on trivial cases (perfect detector → 1.0; zero-precision → 0.0)
  - Both AP methods (continuous + interp) parametrised; interp empty-recall guard
  - `compute_map` strategy divergence: single GT, two competing preds —
    greedy AP=1.0, iou_prior AP=0.5
  - CI bounds within [0,1], lower ≤ estimate ≤ upper
  - Full `Evaluation` round-trip: check `metrics` keys, `cm` shape,
    and that `ap50` field is populated for each class
  - NMS: `filter_by_confidence` drops rows below threshold; `apply_nms`
    suppresses same-class containment and cross-class IoU duplicates

Run: `uv run pytest --cov=src/metrics tests/`

---

## Local Evaluation

`scripts/eval.py` runs a full end-to-end evaluation against the fixture data:

```bash
uv run python scripts/eval.py
```

What it does:
1. Loads `fixtures/ground_truths_all.csv` and `fixtures/predicts_all.csv`
2. Applies predictions preprocessing (confidence filter + NMS) — affects P/R/F1/CM only
3. Calibrates confidence thresholds on the `val` split
4. Evaluates on the `test` split
5. Writes results to `fixtures/` and prints a per-class summary table

Current settings in the script. `matching_strategy`/`ap_method` are pinned to
the documented baseline because the `Evaluation` defaults are now YOLO-like
(`iou_prior`/`interp`):
- `iou_threshold=0.3`
- `matching_strategy="greedy"`, `ap_method="continuous"` (pinned)
- `preprocess_preds_conf_threshold=0.1`
- `preprocess_preds_nms_containment_threshold=0.9`
- `preprocess_preds_nms_iou_threshold=0.6`

Baseline results (greedy strategy, continuous AP method):
- mean P=0.842  R=0.799  F1=0.808  mAP50=0.680  mAP50-95=0.452
- 49 classes, NMS removed 1549/21280 prediction rows

iou_prior strategy results (same preprocessing):
- mAP50=0.674  mAP50-95=0.449  (mean R drops to 0.717)

---

## What Must Be Preserved

All of the following must exist after any refactor:

- `Evaluation(preds_df, split_df, iou_threshold, preprocess, skip_cohen_kappa,
  matching_strategy, preprocess_preds_conf_threshold,
  preprocess_preds_nms_containment_threshold, preprocess_preds_nms_iou_threshold,
  ap_method, confidence_optimization, weights_path, backend, predict_kwargs)` —
  `preds_df` may be `None` (an empty placeholder is created) to predict first;
  `weights_path` is an optional YOLO weights path. `backend` (`None` = native;
  `"ultralytics"` / `"torchmetrics"`) makes `Evaluation` the single entry point:
  when set, the call scores the split with that external library over
  `_raw_preds_df` instead of the native pipeline (see `evaluation.detection_metrics`
  below). `predict_kwargs` (`dict | None`) is forwarded to Ultralytics'
  `model.predict` when predictions are auto-generated from `weights_path`
  (e.g. `{"conf": 0.25, "imgsz": 1280, "half": True}`); ignored when `preds_df` is
  given. The constructor also accepts three optional **grouped config** objects —
  `scoring` (`ScoringConfig`), `preprocessing` (`PreprocessConfig`), `inference`
  (`InferenceConfig`) — as a tidier alternative to the flat kwargs above. They are
  purely additive: every flat kwarg still works, and when a group is passed it
  supplies that whole group and takes precedence over its corresponding flat kwargs
  (defaults mirror the flat defaults). `backend` stays a flat top-level arg
- `ScoringConfig` / `PreprocessConfig` / `InferenceConfig` exported from `metrics`
  (defined in `config.py`) — `ScoringConfig(iou_threshold, matching_strategy,
  ap_method, confidence_optimization, skip_cohen_kappa)`; `PreprocessConfig(dedup_gt,
  conf_threshold, nms_containment_threshold, nms_iou_threshold)` (`dedup_gt` ← the
  flat `preprocess`); `InferenceConfig(weights_path, predict_kwargs)`
- `evaluation(split, find_best_confs, calibration_split)` — main call. When
  `preds_df` was `None`, the first run generates predictions from `weights_path`
  over just the splits it will use — the evaluation split plus `calibration_split`
  (or every image when `split="all"`), via `_splits_to_predict`; if `preds_df` is
  `None` and no `weights_path` was given it raises `ValueError`. After confidence
  optimisation it logs a warning for any class whose chosen threshold equals the
  minimum prediction confidence (the cut keeps every detection, so optimisation
  had no effect — e.g. identical pred/GT boxes)
- `evaluation.predict_to_dataframe(weights, split, conf, iou, imgsz, device,
  image_name, **model_kwargs)` — optional YOLO inference: runs Ultralytics
  ``weights`` over the images in ``split_df['image_path']`` and stores predictions
  in the standard schema as ``preds_df``/``_raw_preds_df`` (``image_name`` = last
  path part). `split` accepts a single split, a list of splits (e.g.
  `["test", "val"]`), or `None` (every image); selecting splits needs a `"split"`
  column. `**model_kwargs` are forwarded verbatim to `model.predict` (e.g. `half`,
  `augment`, `max_det`, `classes`). Requires the `ultralytics` extra (lazy import,
  `ImportError` with hint); raises `ValueError` when `split_df` has no `image_path`
  column (or no `"split"` column when `split` is given). Row-building helpers in
  `inference/yolo_predict.py` (`predict_on_images`, `_detection_rows`) are
  torch-free (`predict_on_images` also accepts `**model_kwargs`).
- Input validation (raises `ValueError`): missing required columns, `NA` in the
  predictions `confidence` column, prediction labels absent from the GT class
  vocabulary, and val/test calibration splits that share an `image_name`
- `evaluation.compute_metrics_ultralytics(split)` /
  `evaluation.compute_metrics_torchmetrics(split)` — score `split` with that
  external backend over `_raw_preds_df` and return `dict[str, DetectionMetrics]`
  (auto-predicting first via `weights_path` when needed); require the matching
  optional extra (lazy import, `ImportError` with hint)
- `evaluation.metrics` — `dict[str, Metrics]`. In `backend` mode this holds the
  backend's numbers adapted onto native `Metrics`: TP/FP/FN are reconstructed as
  floats from the per-class GT count so precision/recall/f1/AP reproduce the
  backend exactly and the dashboards/CI plots keep working (`cohen_kappa = -1`,
  confidence `0.0`)
- `evaluation.detection_metrics` — `dict[str, DetectionMetrics]`; the raw external
  backend output, populated only in `backend` mode (empty otherwise)
- `evaluation.cm`, `evaluation.class_labels` — populated natively and by the
  `"ultralytics"` backend (Ultralytics' own confusion-matrix logic); `None` for the
  `"torchmetrics"` backend, where `get_dashboards` then skips the CM sheet
- `evaluation.best_confidences` — `dict[str, float]` (per-class optimal threshold)
- `evaluation.unfiltered_matches` — matches before confidence slicing
- `evaluation._raw_preds_df` — unpreprocessed predictions (passed to `compute_map`)
- `evaluation.get_dashboards(save_to_excel, path, save_confusion_matrix)`
- `evaluation.plot_confidence_intervals(metric, confidence_level, save_path, figsize)`
- `evaluation.get_topk_confusions(main_class, k)`
- `evaluation.get_dfs_visualization()`
- `Metrics` fields: `tp, fp, fn, confidence, ap50, ap75, ap50_95, cohen_kappa,
  precision, recall, f1_score, perebrak, nedobrak, *_ci_lower, *_ci_upper`
  — `ap50/ap75/ap50_95` are `float | nan` (NaN when class absent from split)
- `PredictMatch` with `type` computed field and an optional `iou` field (IoU
  with the associated GT box; `None` for `background` FPs and FNs)
- `preprocess.filter_by_confidence(preds_df, threshold)` → filtered DataFrame
- `preprocess.apply_nms(preds_df, same_class_containment_threshold, cross_class_iou_threshold)`
  → DataFrame with suppressed rows removed
- `APMethod = Literal["interp", "continuous"]` exported from `metrics`
- `MatchingStrategy = Literal["greedy", "hungarian", "iou_prior"]` exported from `metrics`
- `ConfidenceOptimization = Literal["per_class", "global"]` exported from `metrics`
  — `"global"` selects a single YOLO-style threshold (max mean per-class F1) shared
  by all classes; `"per_class"` (default) tunes one threshold per class.
  Both pick thresholds only at *realizable* operating points (confidence
  tie-group boundaries), so the optimised F1 equals the F1 actually obtained when
  the threshold is applied via `slice_by_conf` — tied confidences cannot produce
  a non-achievable mid-tie optimum.
- `compute_map(gt_df, preds_df, metrics, split_image_names, method, strategy)` —
  all three strategies supported including `"hungarian"`
- `compute_detection_metrics(gt_df, preds_df, backend, classes, split_image_names)`
  and `Backend = Literal["ultralytics", "torchmetrics"]` exported from `metrics` —
  single entry point dispatching to the two external backends; raises `ValueError`
  on an unknown backend (before any heavy import)
- `DetectionMetrics` exported from `metrics` — shared per-class result model
  (`precision/recall/f1/ap50/ap75/ap50_95`) returned by both backends;
  `YoloMetrics` is kept as a backward-compatible alias of it
- `compute_ultralytics_metrics(gt_df, preds_df, classes, split_image_names,
  conf_threshold)` exported from `metrics` — optional YOLO-comparable P/R/F1/AP
  via Ultralytics' own `ap_per_class`. `conf_threshold` (`None` | `float` |
  `dict[str, float]`) reads P/R/F1 off the per-class curves at a given confidence
  (e.g. one calibrated on val) instead of the in-sample max-F1 point; AP is always
  over the full curve. Requires the `ultralytics` extra (lazy import, raises
  `ImportError` with an install hint when missing)
- `find_ultralytics_confidence(gt_df, preds_df, classes, split_image_names, mode)`
  exported from `metrics` — calibration helper: the F1-optimal confidence on a
  split from Ultralytics' F1-vs-confidence curves. `mode="global"` returns a
  `float` (max mean per-class F1), `"per_class"` a `dict[str, float]`; pair with
  `compute_ultralytics_metrics(conf_threshold=...)`. Requires the `ultralytics` extra
- `compute_ultralytics_confusion_matrix(gt_df, preds_df, classes,
  split_image_names, conf, iou_thres)` exported from `metrics` — optional
  confusion matrix via a numpy port of Ultralytics'
  `ConfusionMatrix.process_batch`; returns `(matrix, labels)` with shape
  `(nc+1, nc+1)` in sklearn row=GT/col=pred orientation; requires the
  `ultralytics` extra (lazy import, raises `ImportError` with an install hint)
- `compute_torchmetrics_metrics(gt_df, preds_df, classes, split_image_names)`
  exported from `metrics` — optional general COCO mAP via torchmetrics'
  `MeanAveragePrecision`; requires the `torchmetrics` extra (lazy import, raises
  `ImportError` with an install hint when missing)
