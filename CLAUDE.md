# CLAUDE.md

## Project Overview

Object detection evaluation library. Computes detection metrics (precision, recall, mAP50/75/50-95,
F1, Cohen's kappa, confidence intervals) from pandas DataFrames of ground-truth and predictions.
Output: per-class `Metrics` objects, confusion matrix, Excel dashboards, CI plots.

---

## Environment

- **Python**: 3.11+
- **Package manager**: `uv` with lockfile (`uv.lock`)
- **Virtual environment**: `uv venv` (`.venv/`)

```bash
uv venv
uv sync          # install from lockfile
uv add <pkg>     # add dependency (updates pyproject.toml + uv.lock)
uv run pytest    # run tests inside venv
```

Never use `pip` directly. Always use `uv`.

---

## Code Style

### Linting & type-checking

```bash
uv run ruff check .          # lint
uv run ruff check . --fix    # lint + autofix
uv run ruff format .         # format
uv run mypy src/             # type-check
```

Ruff config is in `pyproject.toml`. mypy runs in strict mode.

### General principles

- **Simple over clever**: readable code is preferred over micro-optimisations.
  Optimise only where profiling shows a bottleneck.
- **Explicit types**: all public functions must have full type annotations.
- **Small, focused functions**: each function does one thing.
- **No dead code**: remove commented-out blocks, unused imports, TODO stubs.
- **Logging**: use `loguru`. No bare `print()` calls.
- **No mixed languages in logs/comments**: use English throughout.

---

## Project Structure

```
src/
  metrics/
    __init__.py
    types.py          # Pydantic models: PredictMatch, Metrics
    iou.py            # IoU matrix computation
    matching.py       # box matching logic — greedy AND hungarian variants
    ap.py             # AP / mAP computation (YOLO-style VOC 2010+)
    confidence.py     # best-confidence search per class
    kappa.py          # Cohen's kappa (pixel-mask method)
    ci.py             # Wilson confidence interval
    evaluation.py     # Evaluation orchestrator class
    dashboard.py      # Excel export + CI plots
    confusion.py      # confusion matrix helpers
    nms.py            # predictions preprocessing: confidence filter + custom NMS
tests/
  conftest.py
  test_iou.py
  test_matching.py
  test_ap.py
  test_ci.py
  test_evaluation.py
  test_nms.py
pyproject.toml
uv.lock
CLAUDE.md
```

---

## DataFrame Schema

Both DataFrames share these columns:

| Column | Type | Description |
|---|---|---|
| `image_name` | str | Unique image identifier |
| `instance_label` | str | Class name |
| `bbox_x_tl` | float | Bounding box top-left x |
| `bbox_y_tl` | float | Bounding box top-left y |
| `bbox_x_br` | float | Bounding box bottom-right x |
| `bbox_y_br` | float | Bounding box bottom-right y |
| `split` | str | `train` / `val` / `test` (GT only) |
| `confidence` | float | Detection score (predictions only) |

---

## Metric Definitions

### Standard metrics (YOLO-compatible)

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

### mAP (YOLO-identical)

mAP is computed **independently** from the per-threshold matching used for P/R/F1.
It has its own inner loop that re-runs greedy matching for each IoU threshold,
exactly as Ultralytics does:

- Sort all predictions by confidence descending (globally per class).
- For each IoU threshold in `[0.50, 0.55, …, 0.95]` (10 values):
  - Match greedily: each prediction claims the highest-IoU unmatched GT.
  - Accumulate cumulative TP/FP → precision-recall curve.
- **AP** — area under P-R curve, VOC 2010+ interpolation:
  precision envelope (right-to-left max), then sum of rectangle areas at
  recall change points.
- **mAP50** — AP at IoU = 0.50
- **mAP75** — AP at IoU = 0.75
- **mAP50-95** — mean of AP over all 10 thresholds

`_compute_ap` must be byte-for-byte equivalent to
[`ultralytics/utils/metrics.py::compute_ap`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py).

Classes with **no GT instances in the evaluated split** receive `float("nan")`
for `ap50`, `ap75`, and `ap50_95` — not `0.0`. This matches how YOLO reports
mAP50 and allows `nanmean` to correctly exclude absent classes from averages.

### What is NOT identical to YOLO

Precision, recall, F1, and the confusion matrix are computed from
`_match_boxes` which runs once at a single IoU threshold with optional
confidence filtering and label-aware TP classification.
YOLO does not expose per-threshold P/R/F1 in the same way.
These metrics are intentionally custom (they support domain-specific
outputs like *perebrak*, *nedobrak*, confidence-interval bars, and
per-image error auditing). Do not try to make them numerically match
YOLO's console output.

---

## Box Matching — Greedy vs Hungarian

`matching.py` must expose **two** matching strategies behind a common interface:

```python
MatchingStrategy = Literal["greedy", "hungarian"]

def match_boxes(
    gt_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    iou_threshold: float,
    strategy: MatchingStrategy = "greedy",
    split_image_names: list[str] | None = None,
) -> dict[str, list[PredictMatch]]:
    ...
```

`split_image_names` is an internal parameter populated by `Evaluation._call`
from `gt_df["image_name"].unique()` (the split-filtered ground truth).
External callers of `match_boxes` can pass `None` to iterate only over images
present in `gt_df`.

### Greedy (default, YOLO-style)

1. Sort predictions by confidence descending.
2. For each prediction, find the highest-IoU unmatched GT.
3. If IoU ≥ threshold and labels match → TP, mark GT as matched.
4. Otherwise → FP (store closest GT label for confusion matrix).
5. Any unmatched GT → FN.

Greedy is the standard COCO/YOLO protocol. It is fast (O(N·M) per image)
and consistent with how mAP is computed.

### Hungarian (optional, globally optimal)

Uses `scipy.optimize.linear_sum_assignment` on the **negative IoU matrix**
to find the assignment that maximises total IoU across all pred–GT pairs.

Key differences from greedy:
- Assignment is geometry-first, confidence-independent: a low-confidence
  prediction can still claim a GT if it has better geometric overlap.
- In dense scenes with overlapping boxes, avoids the greedy "stealing"
  problem where a high-confidence pred takes a GT away from its best match.
- Produces different (not better or worse by definition) TP/FP/FN counts
  and therefore different P/R/F1 — do **not** expect YOLO-equal numbers
  when using this strategy.
- More expensive: O(N³) per image via the Hungarian algorithm.

**When to use Hungarian**: annotation auditing workflows where you want
to find the globally most-plausible pairing between predicted and GT boxes,
e.g. for the `get_topk_confusions` error-analysis path. The default for
evaluation runs should remain `"greedy"`.

Implementation notes:
- Only pairs with IoU ≥ `iou_threshold` are considered valid matches;
  rejected pairs become FP/FN even if they were part of the optimal assignment.
- After assignment, classify each matched pair as TP (labels match) or FP
  (labels differ); unmatched preds → FP, unmatched GTs → FN.
- Add `scipy` to dependencies if not already present.

---

## Tests

- Use `pytest` + `pytest-cov`.
- Place fixtures in `tests/conftest.py`.
- Cover:
  - IoU edge cases (perfect overlap, no overlap, partial)
  - Greedy matching: hand-computed TP/FP/FN on `tiny_dataset`
  - Hungarian matching: same `tiny_dataset`, verify total TP ≥ greedy TP
    (Hungarian is optimal so it can only do equal or better)
  - AP on trivial cases (perfect detector → 1.0; zero-precision → 0.0)
  - CI bounds within [0,1], lower ≤ estimate ≤ upper
  - Full `Evaluation` round-trip: check `metrics` keys, `cm` shape,
    and that `ap50` field is populated for each class
  - `strategy="hungarian"` path runs without error and returns same keys
  - NMS: `filter_by_confidence` drops rows below threshold; `apply_nms`
    suppresses same-class containment and cross-class IoU duplicates

Run: `uv run pytest --cov=src/metrics tests/`

---

## What Must Be Preserved

All of the following must exist after the refactor:

- `Evaluation(preds_df, split_df, iou_threshold, preprocess, skip_cohen_kappa,
  matching_strategy, preprocess_preds_conf_threshold,
  preprocess_preds_nms_containment_threshold, preprocess_preds_nms_iou_threshold)`
- `evaluation(split, find_best_confs, calibration_split)` — main call
- `evaluation.metrics` — `dict[str, Metrics]`
- `evaluation.cm`, `evaluation.class_labels`
- `evaluation.best_confidences` — `dict[str, float]` (per-class optimal threshold)
- `evaluation.unfiltered_matches` — matches before confidence slicing
- `evaluation.get_dashboards(save_to_excel, path, save_confusion_matrix)`
- `evaluation.plot_confidence_intervals(metric, confidence_level, save_path, figsize)`
- `evaluation.get_topk_confusions(main_class, k)`
- `evaluation.get_dfs_visualization()`
- `Metrics` fields: `tp, fp, fn, confidence, ap50, ap75, ap50_95, cohen_kappa,
  precision, recall, f1_score, perebrak, nedobrak, *_ci_lower, *_ci_upper`
  — `ap50/ap75/ap50_95` are `float | nan` (NaN when class absent from split)
- `PredictMatch` with `type` computed field
- `nms.filter_by_confidence(preds_df, threshold)` → filtered DataFrame
- `nms.apply_nms(preds_df, same_class_containment_threshold, cross_class_iou_threshold)`
  → DataFrame with suppressed rows removed
