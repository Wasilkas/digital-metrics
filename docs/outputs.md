[← Documentation index](README.md) · [🇷🇺 Русская версия](outputs.ru.md)

# Outputs, dashboards and error audit

## `ev.metrics` — `dict[str, Metrics]`

Each `Metrics` object exposes:

| Field | Description |
|---|---|
| `tp`, `fp`, `fn` | True positives / false positives / false negatives |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | 2 · P · R / (P + R) |
| `perebrak` | 1 − precision (false-positive rate) |
| `nedobrak` | 1 − recall (miss rate) |
| `ap50` | AP at IoU = 0.50 (`nan` when class absent from split) |
| `ap75` | AP at IoU = 0.75 (`nan` when class absent from split) |
| `ap50_95` | mAP averaged over IoU 0.50 … 0.95 (`nan` when class absent from split) |
| `cohen_kappa` | Cohen's kappa via pixel-mask method |
| `confidence` | Best confidence threshold for this class |
| `precision_ci_lower/upper` | Wilson 95 % CI on precision |
| `recall_ci_lower/upper` | Wilson 95 % CI on recall |
| `perebrak_ci_lower/upper` | CI on perebrak (1 − precision) |
| `nedobrak_ci_lower/upper` | CI on nedobrak (1 − recall) |

## Dashboards and plots

```python
# Excel dashboards + optional confusion-matrix image
summary_df, detail_df = ev.get_dashboards(
    save_to_excel=True,
    path="/path/to/output/",
    save_confusion_matrix=True,
)

# Confidence-interval bar chart
fig, ax = ev.plot_confidence_intervals(
    metric="precision",         # or "recall"
    confidence_level=0.95,
    save_path="/path/to/ci_plot.png",
)
```

Output directories (`path` / the parent of `save_path`) are created
automatically if they don't exist.

## Error audit

```python
# Top-k prediction/GT pairs confused between two classes
audit_df = ev.get_topk_confusions(main_class="car", k=20)

# DataFrames annotated with match type for visualisation.
# gt_df:    predict_type in {"TP", "FN"}
# preds_df: predict_type in {"TP", "FP"}
gt_vis, pred_vis = ev.get_dfs_visualization()

# apply_thresholds=True folds the per-class best_confidences in (like
# slice_by_conf): predictions below their class threshold become "filtered",
# and a GT detected only by such a filtered prediction turns "FN".
gt_vis, pred_vis = ev.get_dfs_visualization(apply_thresholds=True)
```

