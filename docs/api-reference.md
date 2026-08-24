[← Documentation index](README.md) · [🇷🇺 Русская версия](api-reference.ru.md)

# API reference

```python
Evaluation(
    preds_df: pd.DataFrame | str | None,   # DataFrame, CSV path, or None to predict first
    split_df: pd.DataFrame | str,
    iou_threshold: float = 0.5,
    preprocess: bool = False,       # deduplicate near-identical GT boxes
    skip_cohen_kappa: bool = True,  # kappa is expensive; enable only when needed
    matching_strategy: MatchingStrategy = "iou_prior",  # "iou_prior" | "greedy" | "hungarian"
    preprocess_preds_conf_threshold: float | None = None,
    preprocess_preds_nms_containment_threshold: float | None = None,
    preprocess_preds_nms_iou_threshold: float | None = None,
    ap_method: APMethod = "interp",                              # "interp" | "continuous"
    confidence_optimization: ConfidenceOptimization = "per_class",  # "per_class" | "global"
    weights_path: str | None = None,   # YOLO weights to auto-predict from when preds_df is None
    backend: Backend | None = None,    # None = native; "ultralytics" | "torchmetrics"
    predict_kwargs: dict | None = None,  # extra model.predict(...) kwargs for the weights flow
    transliterated_labels: bool = False, # model predicts transliterated class names
    translit_match_cutoff: float = 0.8,  # fuzzy-match cutoff for the restoration above
)
```

The defaults are YOLO-like (`matching_strategy="iou_prior"`, `ap_method="interp"`).

`preds_df` accepts a DataFrame, a CSV path, or `None`. Pass `None` together with
`weights_path` to run the **whole pipeline from weights** — the first call
generates predictions from the model over just the splits it will use (the
evaluation split plus any `calibration_split`), then evaluates:

```python
ev = Evaluation(None, "ground_truth.csv", weights_path="best.pt")
ev(split="val")   # predicts from best.pt, then evaluates
```

If `preds_df` is `None` and no `weights_path` is given, calling the evaluation
raises `ValueError`. You can still predict manually first via
[`predict_to_dataframe`](inference.md).

The image scope for each split is derived automatically from the `split`
column in `split_df` — no extra list needs to be passed.

`confidence_optimization` — `"per_class"` (default) tunes one threshold per
class; `"global"` picks a single YOLO-style threshold shared by all classes
(see [Confidence-threshold optimization](calibration.md#confidence-threshold-optimization)).

`backend` — `None` (default) runs the native pipeline; `"ultralytics"` /
`"torchmetrics"` make `Evaluation` score the split through that external library
instead (see [`Evaluation` with an external backend](backends.md#evaluation-with-an-external-backend)).

`predict_kwargs` — extra keyword arguments forwarded to Ultralytics'
`model.predict` when predictions are auto-generated from `weights_path` (e.g.
`{"conf": 0.25, "imgsz": 1280, "half": True, "augment": True}`). Ignored when
`preds_df` is provided. For one-off control, pass the same kwargs straight to
[`predict_to_dataframe`](inference.md).

`transliterated_labels` — see
[Transliterated model labels](transliteration.md).

`preprocess_preds_conf_threshold` — drop predictions with `confidence <
threshold` before evaluation.

`preprocess_preds_nms_containment_threshold` — same-class containment
suppression: the lower-confidence box is removed when
`intersection / min(area_a, area_b) >= threshold`.

`preprocess_preds_nms_iou_threshold` — cross-class IoU suppression: the
lower-confidence box is removed when `IoU >= threshold`.

Setting either NMS threshold to `None` disables that suppression type.

## Grouped config objects (optional)

If you'd rather not pass a dozen flat keyword arguments, the constructor also
accepts three optional grouped configs. They are **purely additive** — every flat
kwarg above still works unchanged — and each group, when passed, supplies that
whole group and takes precedence over its corresponding flat kwargs:

```python
from digital_metrics import Evaluation, ScoringConfig, PreprocessConfig, InferenceConfig

ev = Evaluation(
    preds_df,
    split_df,
    scoring=ScoringConfig(iou_threshold=0.5, matching_strategy="greedy"),
    preprocessing=PreprocessConfig(conf_threshold=0.25, nms_iou_threshold=0.5),
    inference=InferenceConfig(weights_path="best.pt", predict_kwargs={"imgsz": 1280}),
)
```

- **`ScoringConfig`** — `iou_threshold`, `matching_strategy`, `ap_method`,
  `confidence_optimization`, `skip_cohen_kappa`.
- **`PreprocessConfig`** — `dedup_gt` (the flat `preprocess`), `conf_threshold`,
  `nms_containment_threshold`, `nms_iou_threshold`.
- **`InferenceConfig`** — `weights_path`, `predict_kwargs`.

Each config's defaults mirror the flat-kwarg defaults, so `Evaluation(preds, split)`
and `Evaluation(preds, split, scoring=ScoringConfig())` behave identically.
`backend` stays a flat top-level argument.

