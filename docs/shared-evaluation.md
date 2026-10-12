# Shared evaluation and comparison

These APIs are additive. Existing `Evaluation`, backend constructors, preprocessing,
configuration, command-line scripts, metric definitions and dashboard APIs retain
existing behavior. The legacy `compute_map` continues to mutate supplied `Metrics`
in place and return `None`, defaulting to continuous integration. `Evaluation`
continues to default to interpolated AP.

## Fixed scoring

```python
from digital_metrics import evaluate_fixed

result = evaluate_fixed(
    ground_truth, prepared_predictions, unfiltered_predictions,
    classes=["defect"], thresholds={"defect": 0.7},
    split_image_names=["image-1", "empty-image"],
    iou_threshold=0.5, matching_strategy="greedy", ap_method="interp",
)
```

The caller supplies scoped, validated ground truth; fixed scoring predictions may
already have confidence filtering or NMS. AP predictions must retain the full
geometry-valid population before that preprocessing. Neither population is
modified. This API applies no calibration, class restoration, source lineage,
model vocabulary eligibility, publication policy or data loading.

`FixedEvaluationResult` exposes `metrics`, thresholded `matches`,
`unfiltered_matches`, `cm`, `class_labels`, `gt_status`, `pred_status`, and
`pr_curves`. Status frames use the original hashable input indices as `gt_index`
and `pred_index`, with `image_name`, `instance_label`, and boolean `detected` or
`is_tp` columns. Empty-image placeholders remain in the image scope and contribute
no ground-truth boxes. Predictions on those images contribute false positives.
Classes outside the supplied vocabulary have no status records. Missing confidence
thresholds retain the existing `slice_by_conf` default of zero. Thresholds must be
finite probabilities. Currently `skip_cohen_kappa=False` raises `ValueError`; the
skipped fixed result retains `compute_metrics_from_matches`'s zero kappa default.
Legacy `NativeEngine` retains its separate skipped-kappa sentinel.

`PRObservation` fields are `class_name`, `recall`, `precision`, `confidence`,
cumulative integer `tp`/`fp`, `gt_count`, `ap50`, and `integration_method`.
`build_pr_observations` exposes the same observations independently. Observations
and `compute_map` share the authoritative assignment and AP computation, including
float32 boxes/count arithmetic, default pandas descending confidence sort and
integration sentinels. The AP loop does not rematch at fixed thresholds. Classes
without scoped ground truth receive empty observations and `ap50=None`, while
legacy metric AP fields remain NaN. Explicit image scope filters both populations
for additive fixed and curve APIs. Legacy compute_map scope behavior is preserved.

## Paired statistics

```python
from digital_metrics import compare_paired

paired = compare_paired(
    baseline.gt_status, baseline.pred_status,
    candidate.gt_status, candidate.pred_status,
    images=["image-1", "empty-image"], classes=["defect"],
    iterations=10_000, seed=0, q=0.05,
)
```

Every delta is candidate minus baseline. The exact two-sided McNemar test pairs
ground-truth indices; image bootstrap resamples shared image positions for both
models and estimates pooled-rate differences. Zero-denominator draws are skipped.
Precision contributes both the bootstrap interval and two-sided centred p-value
(with the requested iteration resolution floor); recall contributes an exact
McNemar p-value and a bootstrap interval with an intentionally unavailable p-value.
These are the unchanged arithmetic and failure behaviors extracted from the
consumer's significance implementation.

`PairedComparisonResult.by_class` maps caller-selected classes to
`PairedClassResult(precision, recall_interval, recall)`. `adjusted` is `BHResult`:
one family ordered by every class's precision, then every class's recall.
Nonfinite p-values are excluded and retain their positions as NaN. `pooled`
contains statistics restricted to the same supplied classes and stays outside
BH. `classes=None` computes only one pooled pair over every supplied observation,
with no per-class results and an empty BH family. An explicit empty class list
returns unavailable pooled tests. Duplicate class entries raise `ValueError`.
The caller owns eligibility, display ordering, exclusions and verdicts. Individual
`mcnemar_recall`, `bootstrap_precision_delta`, `bootstrap_recall_delta`, and
`adjust_benjamini_hochberg` functions are also public. Diagnostic callbacks receive
existing empty-population, out-of-scope-row and degenerate-bootstrap messages.

## Explicit rendering

```python
from pathlib import Path
from digital_metrics import render_evaluation

rendered = render_evaluation(
    result.metrics, ground_truth, result.cm, result.class_labels,
    suffix="test", path="metrics/",
    artifact_paths={"precision": Path("plots/precision-test.png")},
    caption="Model: detector\nTraining task: example",
)
```

Artifact destination keys are `dashboard`, `dtrk_dashboard`, `confusion_matrix`,
`recall`, `precision`, `perebrak`, and `nedobrak`; unknown keys raise `ValueError`.
Unspecified destinations use the existing suffixed filenames. Excel writes obey
`save_to_excel` and `save_confusion_matrix`, and an unavailable confusion matrix
has no written matrix artifact. `RenderedEvaluation` returns dashboard DataFrames,
optional `dashboard_path`, `dtrk_dashboard_path`, `confusion_matrix_path`, and
`plots`. Each `RenderedPlot` returns its actual `path`, original `caption`,
`figure`, and `axes`. Figures are closed after saving and remain inspectable.
Captions wrap at 100 characters, render as literal text above the figure and are
stored in PNG metadata under `Model identity`. Each plot saves exactly once after
annotation, including cleanup on save failure, and no interactive display occurs.
Callers own workbook identity annotation and external artifact publication.

The old `get_dashboards` still returns its two-DataFrame tuple, writes the same
filenames and CI plots, and retains its existing save flags. The old
`plot_confidence_intervals` retains `plt.show()` when no save path is supplied and
its existing saved-figure behavior. Both paths share only table/figure construction.

See the [extraction plan](plans/shared-extraction/plan.md) and
[dated validation evidence](plans/shared-extraction/evidence-2026-10-12.md).
