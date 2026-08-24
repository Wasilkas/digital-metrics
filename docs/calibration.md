[← Documentation index](README.md) · [🇷🇺 Русская версия](calibration.ru.md)

# Confidence calibration

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

