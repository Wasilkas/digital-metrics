# Backend-specific detection matching and AP

Native `Evaluation` supports independent greedy, IoU-prior and Hungarian
assignment. Native `iou_prior` is not an exact Ultralytics validator replica.
Ultralytics uses its validator assignment and AP integration; TorchMetrics uses
COCO confidence-ordered matching, precision envelopes and maxDet limits.
Therefore neither identical inputs nor an identically named integration method
guarantees backend AP parity.

For backend comparisons, score the same complete GT and prediction scope with
the same backend and installed backend version. Confidence filtering and NMS can
change matching, prediction ordering and the resulting AP curve. A threshold
only leaves AP unchanged in the optional backend APIs that explicitly compute AP
from the full prediction set while separately computing retained counts.

```python
from digital_metrics import Evaluation, ScoringConfig, PreprocessConfig

ev = Evaluation(
    preds_df, split_df, backend="ultralytics",
    scoring=ScoringConfig(confidence_optimization="per_class"),
    preprocessing=PreprocessConfig(),
)
ev(split="test", calibration_split="val")
```

Calibration selects an observed confidence and keeps every score `>=` that
threshold. It resolves equal objectives to the highest threshold. Thresholded
P/R/F1 carry observed TP/FP/FN; unthresholded backend summary readouts are not
observed counts. COCO limits retained detections to 100 per image/class; this
limit is not imposed on Ultralytics input DataFrames.

See [the detailed explanation](why_prf1_differs.md) and
[dated verification](review-remediation-2026-10-04.md). The small checked-in CSVs
are synthetic examples; historical dataset aggregate parity claims are not
asserted to be reproducible from them.


Ultralytics `per_class` calibration uses deterministic coordinate ascent on the
realized whole-image macro-F1 when mixed-class IoU ties couple thresholds. Other
thresholds stay fixed during each sweep; candidates are the current class's
observed scores plus the highest dataset score. Sweeps repeat until no coordinate
improves, preferring higher thresholds on equal objectives. This guarantees a
coordinate-local optimum on that grid, without exhaustive joint optimality.
Global scalar calibration and independent COCO per-class calibration remain exact
on their observed-score grids. Cached IoUs and affected-image updates keep the
coordinate search from rematching unrelated images at every candidate.
