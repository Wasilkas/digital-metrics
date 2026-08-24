[← Documentation index](README.md) · [🇷🇺 Русская версия](inference.ru.md)

# YOLO inference

If you have an Ultralytics model rather than a predictions table, run inference
straight from the ground-truth DataFrame — `Evaluation.predict_to_dataframe`
closes the eval pipeline at the front, no `data.yaml` needed:

```python
from digital_metrics import Evaluation

# Ground truth must carry an `image_path` column (full path to each image).
# Construct with preds_df=None, then generate predictions from the model:
ev = Evaluation(None, "ground_truth.csv", iou_threshold=0.5)
ev.predict_to_dataframe("best.pt", split="val")   # fills ev.preds_df
ev(split="val")                                    # evaluate as usual
```

- The image source is `split_df["image_path"]`; `image_name` is the last part of
  that path (`Path(image_path).name`), so predictions join back to the ground
  truth automatically. `instance_label` comes from the model's own class names;
  boxes are pixel `xyxy`.
- `split=` chooses which images to run on: a single split (`"val"`), a list of
  splits (`["test", "val"]`), or `None` for every image in `split_df`. (When
  predictions are auto-generated from `weights_path`, `Evaluation` does this for
  you — running only the evaluation split plus any `calibration_split`.)
- The model runs at `conf=0.001`, `iou=0.7` by default (YOLO val settings) so the
  full precision-recall curve is available downstream; raise `conf=` to pre-filter.
- Any extra `model.predict` arguments pass straight through as keyword arguments —
  `ev.predict_to_dataframe("best.pt", split="val", imgsz=1280, half=True, augment=True)`
  — or, in the auto-predict flow, via the constructor's
  `predict_kwargs={"imgsz": 1280, "half": True}`.
- **GPU memory** — inference runs in chunks of `batch` images (default 16), so
  peak VRAM stays bounded (≈ `batch` × per-image cost) instead of growing with the
  image count. If a run OOMs, lower `batch` first, then `imgsz`, and/or set
  `half=True` — e.g. `predict_kwargs={"batch": 4, "imgsz": 1280, "half": True}`.
  (`batch` is a genuine chunk size here; the Ultralytics `batch` predict kwarg is a
  no-op in streaming mode.)
- `predict_to_dataframe` also **returns** the predictions DataFrame, so you can
  save it (`df.to_csv(...)`) or feed it to
  [`compute_detection_metrics`](backends.md).
- `image_name=` selects the `image_name` format (`"name"` filename+ext, the
  default; `"stem"`; or full `"path"`) — match it to your ground-truth `image_name`.

Requires the `ultralytics` extra (imported lazily; the core install stays
torch-free).

