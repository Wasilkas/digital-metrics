[← Documentation index](README.md) · [🇷🇺 Русская версия](getting-started.ru.md)

# Getting started

```bash
uv pip install git+https://github.com/Wasilkas/digital-metrics
# or
pip install git+https://github.com/Wasilkas/digital-metrics
```

Requires Python 3.11+. The core install is `torch`-free. Optional extras add the
`ultralytics` / `torchmetrics` metrics backends (see
[External metrics backends](backends.md)) and
the `clearml` experiment-tracking layer (see
[Experiment tracking (ClearML)](tracking.md)):

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
- `NA` values in the predictions `confidence` column.

Prediction `instance_label`s absent from the ground-truth class vocabulary are
not an error: they are reported in a warning and dropped before scoring (metrics
are only defined for GT classes). If they are absent only because the model
outputs transliterated names, see
[Transliterated model labels](transliteration.md).

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

