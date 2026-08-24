[← Documentation index](README.md) · [🇷🇺 Русская версия](matching.ru.md)

# Box matching strategies

```python
from digital_metrics import Evaluation, MatchingStrategy

# Default: iou_prior (Ultralytics non-scipy style — IoU-sorted, label-aware)
ev = Evaluation(preds_df, split_df, matching_strategy="iou_prior")

# greedy (YOLO confidence-sorted) or hungarian (globally optimal, geometry-first)
ev = Evaluation(preds_df, split_df, matching_strategy="greedy")
ev = Evaluation(preds_df, split_df, matching_strategy="hungarian")
```

`"iou_prior"` (default) pairs boxes by descending IoU and is label-aware,
mirroring Ultralytics' internal assignment. Use `"greedy"` for confidence-sorted
YOLO-style matching, or `"hungarian"` for annotation-audit workflows where you
want the most plausible pairing between predicted and ground-truth boxes.

