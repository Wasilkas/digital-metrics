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
`ultralytics` / `torchmetrics` metrics backends and the `clearml`
experiment-tracking layer:

```bash
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[clearml]      @ git+https://github.com/Wasilkas/digital-metrics"
```

`clearml` is `torch`-free; the two backends each pull in `torch`.

---

## Quick start

Both DataFrames use the same columns: `image_name`, `instance_label`,
`bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` (+ `split` on ground truth,
`confidence` on predictions). Full schema in
[docs/getting-started.md](docs/getting-started.md).

```python
import pandas as pd
from digital_metrics import Evaluation

preds_df = pd.read_csv("predictions.csv", index_col=0)
split_df = pd.read_csv("ground_truth.csv", index_col=0)

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", calibration_split="val")   # calibrate on val, report on test

for cls, m in ev.metrics.items():
    print(f"{cls}: P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1_score:.3f}  mAP50={m.ap50:.3f}")

print(ev.best_confidences)   # per-class confidence thresholds
print(ev.cm, ev.class_labels)  # confusion matrix
ev.get_dashboards(save_to_excel=True, path="dashboards")
```

---

## Documentation

Full docs live in [docs/](docs/README.md).

**Guides**

- [Getting started](docs/getting-started.md) — install, input schema, validation, quick start
- [Confidence calibration](docs/calibration.md) — val-calibrated thresholds, per-class vs. global
- [Box matching strategies](docs/matching.md) — `iou_prior` / `greedy` / `hungarian`
- [Predictions preprocessing](docs/preprocessing.md) — confidence filter + custom NMS
- [Transliterated model labels](docs/transliteration.md) — restore GT labels from a transliterated vocabulary
- [Reproducing YOLO metrics](docs/yolo-parity.md) — settings that track Ultralytics
- [External metrics backends](docs/backends.md) — `ultralytics` / `torchmetrics`
- [YOLO inference](docs/inference.md) — predictions straight from weights
- [Outputs](docs/outputs.md) — `Metrics`, dashboards, CI plots, error audit
- [Experiment tracking (ClearML)](docs/tracking.md)
- [API reference](docs/api-reference.md) — `Evaluation` constructor, configs, attributes

**Reference**

- [Metric definitions](docs/metrics-definitions.md) — IoU/TP/FP/FN, mAP, AP methods
- [Why P/R/F1 differs from mAP](docs/why_prf1_differs.md) — with plots
- [Changelog](CHANGELOG.md)

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
