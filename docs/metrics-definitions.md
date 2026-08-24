[← Documentation index](README.md) · [🇷🇺 Русская версия](metrics-definitions.ru.md)

# Metric definitions

## Standard metrics (YOLO-compatible)

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

## mAP

mAP is computed **independently** from the per-threshold matching used for P/R/F1.
It uses `_raw_preds_df` (unpreprocessed predictions) and runs its own inner matching
loop for each IoU threshold, mirroring the Ultralytics two-path design:

- Sort all predictions by confidence descending (globally per class).
- Compute each image's pred↔GT IoU matrix once per class (`_precompute_image_matches`)
  and reuse it across all ten thresholds — the IoU is threshold-independent, so this
  avoids ~10x redundant IoU work (the assignment kernel still runs per threshold).
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

## AP Methods (`APMethod`)

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

## What is NOT identical to YOLO

Precision, recall, F1, and the confusion matrix are computed from `match_boxes`
which runs once at a single IoU threshold with optional confidence filtering and
label-aware TP classification. YOLO does not expose per-threshold P/R/F1 in the
same way. These metrics are intentionally custom. Do not try to make them
numerically match YOLO's console output.

