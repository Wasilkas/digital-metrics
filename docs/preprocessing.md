[← Documentation index](README.md) · [🇷🇺 Русская версия](preprocessing.ru.md)

# Predictions preprocessing

Apply confidence filtering and/or custom NMS before evaluation by passing
thresholds to the constructor:

```python
ev = Evaluation(
    preds_df,
    split_df,
    # Drop low-confidence predictions
    preprocess_preds_conf_threshold=0.25,
    # Suppress same-class box that is largely inside another (containment >= 0.8)
    preprocess_preds_nms_containment_threshold=0.8,
    # Suppress lower-confidence box when two different-class boxes overlap (IoU >= 0.5)
    preprocess_preds_nms_iou_threshold=0.5,
)
```

Each threshold is independent — set only the ones you need. Setting a threshold
to `None` (the default) disables that suppression type.

