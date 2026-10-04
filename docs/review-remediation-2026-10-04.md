# Detection metrics review remediation — 2026-10-04

This ledger records the 35 historical review findings and the contracts selected
for their remediation. Historical proof files remain local; this published ledger
and permanent tests are self-contained. The imported unfinished candidate failed
ordinary package import with `NameError: Task` in a TYPE_CHECKING-only annotation.
Quoted forward annotations repaired that import without future annotations.
Candidate changes were inspected and revised rather than accepted as evidence.

The index contract preserves the existing unmatched `-1` sentinel. Supported row
labels are unique, nonmissing hashable labels; duplicate labels and sentinel
collisions are input errors. Box assignment positions remain internal integers.
Empty-image GT rows have missing labels and all coordinates missing. Labelled
partial, inverted or zero-area boxes are malformed. Data are validated before dropping or suppression.

Explicit backend thresholds describe observed retained detections, including
zero TP/nonzero FP and classes absent from a selected GT split. AP remains native
to each backend. Unthresholded summaries retain compatibility reconstruction but
serialize `counts_observed=False`; their Wilson intervals are NaN. Calibration
uses complete observed-score tie groups and the highest equal optimum. COCO's
100-detection limit is per image/class; Ultralytics has no imposed COCO cap.

| Finding | Failing behavior | Decision and code | Permanent acceptance evidence |
|---|---|---|---|
| 001 | Mutable CI bounds stale after count changes | types.py reads current counts on each access | test_mutable_ci |
| 002 | Requested plot coverage only changed its title | dashboard.py recomputes Wilson coverage from current observed counts | test_plot_coverage |
| 003 | Empty dashboard raised KeyError | dashboard.py returns schema-preserving empty tables | test_empty_dashboard_schema |
| 004 | NumPy minimum lacked trapezoid | scoring/ap.py uses scipy.integrate.trapezoid | test_public_ap_validation; minimum full-suite run |
| 005 | Evaluation input fixtures missing | Synthetic input CSVs and create_review_examples.py restore a runnable example, without claiming original provenance | scripts/eval.py actual run |
| 006 | Comparison figures missing | Replace absent original figures with reproducible illustrative SVG | create_review_examples.py and local-link validation |
| 007 | Excel interpreted class labels as formulas | dashboard.py forces literal string cells for labels and CM headers | test_excel_literal_labels_and_suffix |
| 008 | Plot files ignored the dashboard suffix | dashboard.py namespaces every CI plot; tracker uses the same suffix | test_excel_literal_labels_and_suffix; test_log_evaluation_dispatches_all_layers |
| 101 | background collided with unmatched class sentinel | validation.py rejects reserved background vocabulary | test_background_reserved |
| 102 | Wrong-class CM event counted twice | confusion.py claims each GT once; duplicate detections become background FP events | test_wrong_class_one_cm_event |
| 103 | Malformed labelled GT affected recall and AP differently | validation.py rejects incomplete/nonfinite labelled boxes before scoring | test_raw_validation_before_preprocessing; test_nan_gt_row_does_not_misattribute_match |
| 104 | Raster negative slices wrapped around | kappa.py clips all four corners to image bounds | test_kappa_clipping |
| 105 | Package import suppressed unrelated RuntimeWarnings | kappa.py removes global suppression | test_warning_filters_are_not_changed_by_import |
| 106 | Numeric image IDs disappeared | Shared normalization includes GT, predictions and explicit image scope; collisions rejected | test_numeric_ids_and_collision |
| 107 | Zero-trial CI bypassed method checks | ci.py validates method before empty return | test_ci_rejects_invalid_method_even_empty |
| 109 | Closest cross-class GT was not selected | matching.py masks same-class candidates before argmax | test_matching.py cross-class cases |
| 110 | Valid pandas row labels rejected | Hashable row labels survive matching/evaluation/filtering/visualization; internal assignments use integer positions. Reject duplicate/missing labels and -1 collisions | test_preserve_index; test_reject_missing_or_sentinel_row_label; test_label_preservation_through_filter_and_visualization |
| 111 | Prefilter removed invalid confidences before checks | Evaluation validates raw input before preprocessing, unknown-label drops and after generated predictions | test_raw_validation_before_preprocessing |
| 113 | Invalid CI coverage returned invalid bounds | ci.py validates finite coverage strictly inside (0,1), plus finite valid counts | test_ci_rejects_invalid_levels_even_empty |
| 201 | Manual generated predictions could not be evaluated | Evaluation reuses inferred paths independently of weights_path, filling only missing paths | test_yolo_predict.py manual and repeated split tests |
| 202 | Visualization annotated other splits | Evaluation scopes prediction visualization to evaluated image IDs | test_split_visualization_and_dedup |
| 204 | Placeholder-only / all-empty evaluation failed | Explicit background-only CM and schema-preserving empty outputs; placeholders keep image scope | test_empty_evaluation_and_split_validation; backend empty-image oracles |
| 205 | Invalid modes/thresholds silently chose fallbacks | ScoringConfig, PreprocessConfig, public AP/matching/calibration helpers validate options and finite bounds | test_config_validation; test_public_ap_validation |
| 206 | Invalid split selectors leaked internal errors | Evaluation validates available splits/required split column before inference/scoring | test_empty_evaluation_and_split_validation |
| 207 | Image IDs crossed dataset splits | validate_split_ownership requires one split for each normalized ID | test_empty_evaluation_and_split_validation |
| 208 | Fuzzy transliteration bypassed known ambiguity | translit.py retains ambiguous keys and prevents their fuzzy fallback | test_ambiguous_translit_stays_unresolved |
| 209 | Adapted metrics lost calibrated thresholds | BackendEngine copies chosen best_confidences into every Metrics.confidence | test_evaluation_backend.py calibrated threshold tests |
| 210 | GT dedup deleted different-class coincident boxes | Evaluation dedups only within image and class | test_split_visualization_and_dedup |
| 211 | Missing prediction labels leaked KeyError | Raw schema validation precedes label access and filtering | test_missing_label_column |
| 301 | TorchMetrics minimum lacked required compatibility | Raise optional lower bound to actually tested non-yanked 1.3.1; 1.1/1.2 need undeclared setuptools on Python3.12 | Actual 1.3.1 optional suite without setuptools |
| 302 | Thresholded COCO curves omitted retained FPs | operating_points.py computes realizable raw counts independently of AP envelope; explicit threshold uses >=; calibration preserves ties and selects highest equal optimum | test_coco_actual_events; test_calibration_preserves_empty_image_scope_and_ties; test_exact_zero_tp_backend_adapter |
| 303 | Failed tracker context marked complete | Tracker independently detaches logging and closes before fresh-handle failure marking; preserves original exception and avoids unsafe marking after close failure | test_clearml_tracker.py lifecycle tests |
| 304 | GPU cache cleanup retained final inference result | yolo_predict.py releases results, r and boxes before model/allocator cleanup on both success and failure | test_yolo_predict.py; weak-reference lifecycle regressions |
| 305 | False universal backend AP parity claims | Replace same-curve explanation with backend-specific matching/integration/limits; observed thresholded counts separate from native backend summaries | test_ultralytics_actual_validator; test_coco_actual_events; why_prf1_differs.md |
| 306 | Unknown explicit backend classes crashed COCO | prepare_inputs consistently warns/drops predictions outside requested vocabulary | test_torchmetrics_metrics.py; test_ultralytics_metrics.py |

## Verification and scope

Core verification uses `uv run --no-sync pytest`, Ruff and strict mypy. Modern
backend tests use real Ultralytics BaseValidator and pycocotools COCOeval event
arrays as independent oracles for empty-image FPs, zero TP, exact IoU/confidence
boundaries, ties and detection limits. Optional dependency minimum and NumPy
minimum are tested in isolated environments; final results are appended once
complete. Context7 TorchMetrics documentation was consulted; actual compatibility
runs govern the selected lower bound.

Synthetic fixtures contain one calibration positive, one evaluation positive and
one evaluation negative image. Historical generated output CSV/JSON files are
preserved as historical artifacts; they are not claimed to originate from this
small fixture. The reproducible SVG illustrates a synthetic TP/FP/TP sequence.
No original source-data provenance is invented.

Inference lifecycle acceptance establishes Python object release ordering using
fakes. It does not measure CUDA reserved-memory reduction or perform real model
inference. ClearML lifecycle checks include injected SDK-compatible fakes and an owned
live SDK task: the original RuntimeError object was retained and the final remote
status was failed after close/fetch/mark_failed. No production task was changed.

## Recorded checks

The core suite before final positive-area regressions and restored empty-image
matcher coverage passed 280 tests (22 skips). The final affected validation,
preprocessing, matching and review subset passed 93 tests.
The modern optional backend subset with equal-IoU oracle cases passed 27
tests; after adding reverse row-order tie permutations, all 15 operating-point
oracle tests passed. Strict mypy passed for all 36 source files; repository Ruff and whitespace
checks passed. Changed Markdown local links resolved. The synthetic evaluation
example executed successfully. Minimum TorchMetrics 1.3.1 passed its 11-test
backend/calibration subset without setuptools; 1.1 and 1.2 require undeclared
setuptools under Python 3.12. Final combined gates and minimum NumPy verification
are recorded by the parent integration run.


## Independent review corrections

The fresh review reproduced three residual issues: global calibration rematched
all images at every score; COCO retained matching used float64 boxes/scores unlike
the float32 AP adapter; missing unthresholded class summaries claimed observed
zero counts. Calibration now caches per-image/class IoUs and updates only changed
groups at complete original-score events, maintaining exact rational F1 objectives
and highest equal thresholds. COCO retains by original confidence but matches
float32 boxes in stable float32 score order. Missing summaries are explicitly
unobserved with unavailable Wilson intervals. The operation-count regression
requires one rematch per independent image, and boundary oracles cover coordinate
rounding and score ties introduced by the actual adapter's float32 tensors.


Correction verification on 2026-10-04: the final core suite before the last
absent-class acceptance addition passed 291 tests with 32 optional-dependency
skips. The modern backend, calibration and oracle subset passed 43 tests,
including float32 coordinate, xywh conversion and score-order boundaries.
Repository Ruff, strict mypy (36 source files), whitespace and changed Markdown
local-link checks passed. On the same independent-image probe, calibration at
50/100/200 images took 0.033/0.041/0.055 seconds after correction, compared with
0.726/3.001/11.277 seconds before correction. These timings are local observations;
the permanent scaling gate checks bounded IoU and rematch operation counts.

## Parent acceptance — 2026-10-04

- Core final suite: 292 passed, 32 optional skips; Ruff and strict mypy (37 files) passed.
- Modern installed optional backends: complete suite 324 passed, 5 skips. Actual Ultralytics
  validator and COCO event-array oracles cover confidence/IoU boundaries, float32 score ties,
  coordinate conversion, detection limits, empty-image FPs and exact-count adaptation.
- Supported NumPy minimum 1.24 on Python 3.11: complete suite 292 passed, 32 optional skips.
- Minimum TorchMetrics 1.3.1 without setuptools: 41 backend/calibration/oracle tests passed.
- YOLO integration kept committed dependency pins unchanged in one environment and installed
  both candidate dependencies in another. Both full application suites passed 917 tests with
  9 optional skips. Native CPU and single-GPU training/validation/metrics/comparison/reporting
  completed with both dependency configurations; all published artifacts and output models
  were force-downloaded and inspected. Standalone candidate validation/comparison/reporting
  also completed. All owned ClearML tasks, models, artifacts and the disposable project were
  cleaned up, with zero owned tasks/projects remaining.

Parent inspection added a regression for SDK close failure: the watchdog may still be active,
so issuing a failed transition can self-abort the process. The test first failed; the tracker
now reports the close error and preserves the original exception without that unsafe transition.
Independent sink detachment and closure are still attempted. A terminal remote status cannot be
guaranteed when SDK closure fails. This narrow correction is checked separately from the full
suite results above. No actual close failure or measured CUDA memory improvement is claimed.

The original unmatched index sentinel remains `-1`. Intentional compatibility changes reject
malformed raw boxes/identities/options, expose unavailable summary intervals, use realizable
threshold operating points, and suffix confidence plots. AP remains backend-specific. Physical
multi-GPU and Windows outcomes were not exercised. Synthetic fixture provenance is explicit;
original unavailable input data and bulky historical audits are not required by published links.


### Whole-image Ultralytics tie correction

A second fresh review supplied a mixed-class, equal-IoU image for which separate
class matching produced class `1` counts `(2,3,0)`, while the actual whole-image
BaseValidator produced `(1,4,1)`. Although class masks exclude cross-class pairs,
the validator's full match-array tie ordering depends on the other classes.
Ultralytics now caches complete images, preserves mixed-class prediction/GT row
order, applies per-class retention thresholds, and rematches each changed image
jointly. Calibration updates all class counts affected by that image. COCO still
caches image/class groups and preserves its separate score-order semantics.
Permanent actual-validator oracles cover the supplied image, nonuniform threshold
dictionaries, both calibration modes and reversed prediction/GT row orders.
The bounded independent-image rematch regression covers both backends.


Whole-image correction verification on 2026-10-04: full core passed 293 tests
with 40 optional-dependency skips. The modern backend/oracle/calibration/scaling
subset passed 52 tests; the focused core subset passed 62 with 6 optional skips.
Repository Ruff, strict mypy (36 source files), whitespace and changed Markdown
local-link checks passed. Parent-owned full modern, NumPy-minimum and minimum
TorchMetrics gates plus fresh independent review follow this frozen source.


Final parent gates after the whole-image correction: modern complete suite **333 passed,
5 skipped**; NumPy 1.24/Python 3.11 complete suite **293 passed, 40 optional skips**;
TorchMetrics 1.3.1 without setuptools **49 passed** across backend, calibration, scaling
and oracle modules. Ruff and mypy (37 source/release-tool files) passed. Wheel and sdist
rebuilt; an isolated installed wheel imported successfully and passed the reduced mixed-class
Ultralytics regression. Changed Markdown parsing/local links and whitespace checks passed.
The final YOLO pinned and candidate dependency suites each passed **927 tests, 9 optional skips**.


### Realized per-class dictionary refinement

The next fresh oracle reproduced a mixed-class calibration dictionary whose
realized F1 sum was `7/5`, while changing one threshold to the shared `0.9`
dictionary gave `3/2`. Combining independently selected points from scalar sweeps
therefore failed the whole-image realization contract. Ultralytics per-class
calibration now refines its actual dictionary with deterministic coordinate
ascent on exact rational macro-F1. It fixes other thresholds, evaluates each
class's observed scores plus the highest dataset score, rematches only affected
images at score events, and repeats until no coordinate improves. Equal-objective
moves only increase thresholds, ensuring termination on the finite grid. Class
ordering follows the fixed vocabulary; the best scalar dictionary may seed the
search when it improves the realized objective. Cached IoUs are reused and
refinement is skipped when images contain no mixed prediction classes.

This is a coordinate-local optimum, without an exhaustive joint global-optimum
claim. Exact scalar global and independent COCO per-class searches retain their
contracts. Actual-validator tests exercise the returned dictionary, alternate
class/row orders, every single-coordinate candidate, and highest equal ties;
operation-count tests cover independent and mixed-image scaling.


Dictionary-refinement verification on 2026-10-04: the modern backend/oracle/
calibration/scaling subset passed 60 tests. Focused core passed 63 tests with
9 optional-dependency skips. Ruff, strict mypy (36 source files), whitespace and
changed Markdown local links passed. An independent parent-prepared probe of
100 seeded mixed-class cases against actual BaseValidator passed exact returned
counts, coordinate-local macro-F1 optimality and highest equal-threshold checks.
No full core or dependency-minimum result is inferred from these focused checks;
parent-owned final gates and fresh review follow the frozen source.

The parent additionally reproduced an empty-evaluation tracker regression:
stale same-suffix CI files were uploaded despite unavailable current metrics.
CI image upload now requires nonempty evaluation metrics while confusion-matrix
reporting remains available. The parent-owned affected tracker/review subset
passed 50 tests, with Ruff and strict mypy passing.


## Final acceptance and independent review

After dictionary refinement and the empty-tracker correction, parent full modern-backend
verification passed **342 tests, 5 skips**. The supported NumPy 1.24/Python 3.11 environment
passed **295 tests, 47 optional skips**. A missing optional-dependency guard in the new oracle
was corrected after the minimum environment exposed it; the full suite was then rerun.
TorchMetrics 1.3.1 passed **57 tests** in a fresh isolated package path with explicit assertions
that setuptools, pkg_resources and distutils were absent. The earlier shared dependency path
made setuptools importable; this final check supersedes its absence claim.

The built wheel and sdist were refreshed. The installed wheel passed 100 seeded actual-validator
mixed-class probes for exact counts, coordinate-local macro-F1 optimality and highest equal
thresholds. A further 25 seeded cases with three images and three classes passed the same
independent oracle checks. The final proposed-dependency YOLO suite passed **927 tests,
9 optional skips**, matching the independently verified pinned configuration.

A fresh independent Sol review returned **ship**, with no actionable blockers after full source,
test and documentation inspection; its independent focused suite passed **96 tests**, Ruff and
whitespace checks passed. The coordinate-local guarantee and runtime dependence on refinement
sweeps remain explicit. Native/live and platform limits above remain unchanged.
