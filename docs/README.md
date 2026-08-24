[🇷🇺 Русская версия](README.ru.md)

# digital-metrics documentation

Full documentation for [`digital-metrics`](../README.md).

## Guides

| Document | Contents |
|---|---|
| [Getting started](getting-started.md) | Installation and extras, input DataFrame schema, validation rules, quick start |
| [Confidence calibration](calibration.md) | Val-calibrated thresholds, per-class vs. global confidence optimization |
| [Box matching strategies](matching.md) | `iou_prior` / `greedy` / `hungarian` and when to use each |
| [Predictions preprocessing](preprocessing.md) | Confidence filtering and custom NMS before scoring |
| [Transliterated model labels](transliteration.md) | Restoring Cyrillic GT labels from a transliterated model vocabulary |
| [Reproducing YOLO metrics](yolo-parity.md) | Settings that make the native pipeline track Ultralytics |
| [External metrics backends](backends.md) | `ultralytics` / `torchmetrics` backends and `Evaluation(backend=...)` |
| [YOLO inference](inference.md) | Generating predictions from weights with `predict_to_dataframe` |
| [Outputs](outputs.md) | `Metrics` fields, Excel dashboards, CI plots, error audit |
| [Experiment tracking (ClearML)](tracking.md) | Mirroring an evaluation into a ClearML task |
| [API reference](api-reference.md) | `Evaluation` constructor, grouped configs, attributes after a call |

## Reference

| Document | Contents |
|---|---|
| [Metric definitions](metrics-definitions.md) | IoU / TP / FP / FN, mAP, AP integration methods, what does not match YOLO |
| [Why P/R/F1 differs from mAP](why_prf1_differs.md) | Operating point vs. COCO envelope, with plots |
| [Changelog](../CHANGELOG.md) | Release history |
