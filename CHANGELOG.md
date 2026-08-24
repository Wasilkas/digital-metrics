# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Documentation split out of the two READMEs into `docs/` (EN + RU), with the
  READMEs reduced to install, quick start and links.
- Added this changelog.

## [0.5.2] — 2026-08-21

### Fixed

- `NA` ground-truth labels (rows describing empty images) are kept out of Cohen's
  kappa and out of both external metrics backends, so they no longer leak into the
  class vocabulary or the pixel-mask kappa computation.

## [0.5.1] — 2026-08-11

### Fixed

- `NA` ground-truth labels from empty-image rows are ignored when building the
  class list, instead of producing a spurious class.

## [0.5.0] — 2026-08-11

### Added

- Transliterated model labels: `Evaluation(transliterated_labels=True,
  translit_match_cutoff=0.8)` maps a model vocabulary such as `"Gryaz_na_osnove"`
  back onto the Cyrillic ground-truth labels, which would otherwise score all
  zeros. Matching runs forward off the ground truth over three schemes
  (`common` / `gost` / `icao` plus a `ё`→`е` fold), by exact normalised key and
  then by `difflib` fuzzy match; unmatched labels are warned about and left as is.
- New module `translit.py` with public helpers `transliterate(text, scheme)`,
  `restore_labels(labels, gt_labels, cutoff)` and `TRANSLIT_SCHEMES`.
- Label restoration runs before unknown-class dropping on the native path and on
  both backend paths, rewriting `preds_df` and `_raw_preds_df`.

## [0.4.0] — 2026-07-22

### Added

- `get_dfs_visualization(apply_thresholds=True)` folds the per-class
  `best_confidences` into the returned frames (mirroring `slice_by_conf`):
  predictions below their class threshold become `filtered`, and a ground-truth
  box detected only by such a filtered prediction turns `FN`.

## [0.3.2] — 2026-07-02

### Fixed

- ClearML task lifecycle is managed for both entry points (an injected task and a
  tracker-created one), so tasks are closed correctly in either case.

## [0.3.1] — 2026-07-02

### Fixed

- YOLO inference VRAM is bounded by chunking: `predict_on_images` runs
  `model.predict` in chunks of `batch` (default 16) images instead of streaming
  the whole list, whose retained per-image GPU tensors made peak VRAM grow with
  the image count and OOM on large sets. `batch` is the memory lever and also
  flows through `predict_kwargs`.

## [0.3.0] — 2026-07-02

### Added

- Optional ClearML experiment-tracking layer (`digital-metrics[clearml]`,
  `torch`-free, lazily imported): `ClearMLTracker` mirrors a finished
  `Evaluation` into a task — per-class scalars, a metrics table and `mean_*`
  values, dashboard/threshold/confusion-matrix artifacts and the written `.xlsx`
  files, the CI plots plus the confusion-matrix plot, and run logs through a
  `loguru` sink. `summarize_metrics(metrics)` is the torch/ClearML-free helper
  behind it. The core evaluation code has no ClearML dependency.

## [0.2.0] — 2026-06-25

### Changed

- **Breaking**: the import package was renamed `metrics` → `digital_metrics`.
  Update imports to `from digital_metrics import ...`.

## [0.1.0] — 2026-06-05 … 2026-06-24

Initial release and the development that followed on it.

### Added

- `Evaluation` orchestrator computing per-class precision, recall, F1,
  mAP50 / mAP75 / mAP50-95, Cohen's kappa, Wilson confidence intervals, a
  confusion matrix, Excel dashboards and CI plots from ground-truth and
  prediction DataFrames.
- Matching strategies `greedy` (YOLO confidence-sorted), `iou_prior`
  (Ultralytics non-scipy style, now the `Evaluation` default) and `hungarian`
  (globally optimal); shared box-assignment kernels let `compute_map` use all
  three.
- AP integration methods `interp` (101-point COCO, the `Evaluation` default) and
  `continuous` (VOC 2010+ rectangle areas), selected with `ap_method`.
- YOLO-style global confidence optimization (`confidence_optimization="global"`,
  one threshold maximising mean per-class F1) alongside the per-class default.
- Predictions preprocessing before scoring: confidence filtering and custom NMS
  (same-class containment + cross-class IoU); mAP still scores the raw
  predictions, mirroring Ultralytics.
- External metrics backends behind one entry point,
  `compute_detection_metrics(backend=...)`: `ultralytics` (its own `ap_per_class`
  plus a ported confusion matrix) and `torchmetrics` (COCO mAP via
  `MeanAveragePrecision`). Both are optional `torch` extras imported lazily.
  `Evaluation(backend=...)` scores a split with either and adapts the result onto
  native `Metrics` so dashboards and CI plots keep working.
- Calibration for both backends: `find_ultralytics_confidence` /
  `find_torchmetrics_confidence` pick the F1-optimal confidence on a calibration
  split, and `conf_threshold=` reads the eval split's P/R/F1 at that point off the
  per-class curves while AP stays over the full curve — enabling "calibrate on
  val, report on test" outside the native path.
- YOLO adapter: `Evaluation.predict_to_dataframe(...)` runs Ultralytics weights
  over `split_df["image_path"]` and stores predictions in the standard schema;
  `Evaluation(preds_df=None, weights_path=...)` auto-predicts the splits it needs,
  with `predict_kwargs` forwarded to `model.predict`.
- Optional grouped config objects `ScoringConfig` / `PreprocessConfig` /
  `InferenceConfig` as an additive alternative to the flat constructor kwargs.
- Input validation raising `ValueError` on missing columns, `NA` prediction
  confidence and calibration splits sharing an `image_name`; predictions whose
  class is absent from the ground truth are warned about and dropped instead.
- Repository documentation: Russian README, a guide to reproducing YOLO
  (Ultralytics) metrics, comparison/profiling scripts and the fixture evaluation
  script `scripts/eval.py`.

### Changed

- Defaults moved to the YOLO-like combination (`iou_prior` matching, `interp` AP).
- `split_image_names` was removed from the public `Evaluation` API; it is derived
  internally from the split's ground truth.
- Modules regrouped into role subpackages (`matching/`, `scoring/`,
  `preprocess/`, `reporting/`, `backends/`, `inference/`, `engines/`) with a flat
  public API; scoring split into `NativeEngine` / `BackendEngine`, and
  `ConfidenceCalibrator` / `PredictionPreprocessor` extracted from `Evaluation`.

### Fixed

- Classes absent from the evaluated split report `NaN` for `ap50` / `ap75` /
  `ap50_95` instead of `0.0`, so `nanmean` excludes them.
- `compute_map` scores only the current split's images.
- False positives on images with no ground truth are no longer silently dropped
  from mAP and P/R/F1.
- Confidence-tie bug: thresholds are chosen only at realizable operating points
  (tie-group boundaries), so the optimised F1 equals the F1 obtained when the
  threshold is applied.
- Match desync on `NaN` / empty-image rows, and dashboard output directories are
  created when missing.
- `pyproject.toml` dependencies were nested under `[project.urls]`.

### Performance

- Per-image numpy row grouping shared across the pipeline.
- Per-image IoU cached across the ten mAP thresholds (~2.1x faster `compute_map`).
