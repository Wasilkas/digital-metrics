[← Documentation index](README.md) · [🇷🇺 Русская версия](backends.ru.md)

# External metrics backends

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
[docs/why_prf1_differs.md](why_prf1_differs.md) for the explanation and plots.

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

