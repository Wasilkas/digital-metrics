[← Documentation index](README.md) · [🇷🇺 Русская версия](yolo-parity.ru.md)

# Reproducing YOLO metrics

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

## Why other parameters are not a problem

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

