"""Scientific/API/artifact contracts for the additive shared extraction."""

from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from digital_metrics import (
    Metrics,
    adjust_benjamini_hochberg,
    build_pr_observations,
    compare_paired,
    evaluate_fixed,
    render_evaluation,
)
from digital_metrics.comparison import (
    bootstrap_precision_delta,
    bootstrap_recall_delta,
    mcnemar_recall,
)
from digital_metrics.engines import compute_metrics_from_matches
from digital_metrics.matching import match_boxes
from digital_metrics.reporting import get_dashboards
from digital_metrics.scoring import compute_ap, compute_map, get_confusion_matrix, slice_by_conf


@pytest.mark.parametrize("strategy", ["greedy", "iou_prior", "hungarian"])
@pytest.mark.parametrize("method", ["continuous", "interp"])
def test_fixed_population_and_authoritative_observations(tiny_dataset, strategy, method) -> None:
    truth, raw = tiny_dataset
    prepared = raw[raw["confidence"] >= 0.8].copy()
    classes = ["class_a", "class_b", "class_c", "absent"]
    thresholds = {name: 0.9 for name in classes}
    originals = [frame.copy() for frame in (truth, prepared, raw)]
    result = evaluate_fixed(
        truth,
        prepared,
        raw,
        classes=classes,
        thresholds=thresholds,
        matching_strategy=strategy,
        ap_method=method,
    )
    matches = match_boxes(
        truth, prepared, 0.5, strategy=strategy, split_image_names=["img1", "img2"]
    )
    sliced = slice_by_conf(matches, classes, thresholds)
    old_metrics = compute_metrics_from_matches(sliced, classes, thresholds)
    compute_map(truth, raw, old_metrics, ["img1", "img2"], method=method, strategy=strategy)
    for name in classes:
        assert result.metrics[name].model_dump(exclude={"cohen_kappa"}) == pytest.approx(
            old_metrics[name].model_dump(exclude={"cohen_kappa"}), nan_ok=True
        )
        assert result.metrics[name].cohen_kappa == 0
        curve = next(c for c in result.pr_curves if c.class_name == name)
        if curve.gt_count:
            assert curve.ap50 == result.metrics[name].ap50
            assert compute_ap(
                np.asarray(curve.recall), np.asarray(curve.precision), method
            ) == pytest.approx(curve.ap50)
        else:
            assert curve.ap50 is None and curve.recall == []
    assert result.matches == sliced and result.unfiltered_matches == matches
    matrix, labels = get_confusion_matrix(sliced, classes)
    np.testing.assert_array_equal(result.cm, matrix)
    assert result.class_labels == labels
    assert result.gt_status["detected"].sum() == 2
    assert result.pred_status["is_tp"].sum() == 2
    assert len(result.pred_status) == 2
    assert result.pr_curves[0].tp[-1] == 3  # AP uses removed fixed predictions too.
    for frame, original in zip((truth, prepared, raw), originals, strict=True):
        pd.testing.assert_frame_equal(frame, original)


def test_empty_image_false_positive_and_absent_class(tiny_dataset) -> None:
    truth, preds = tiny_dataset
    placeholder = pd.DataFrame(
        [{column: ("empty" if column == "image_name" else None) for column in truth.columns}],
        index=[99],
    )
    truth = pd.concat([truth, placeholder])
    extra = preds.iloc[[0]].copy()
    extra.index = [90]
    extra["image_name"] = "empty"
    extra["confidence"] = 0.99
    preds = pd.concat([preds, extra])
    result = evaluate_fixed(truth, preds, preds, classes=["class_a", "absent"], thresholds={})
    assert result.metrics["class_a"].fp == 3
    assert result.pr_curves[0].fp[0] == 1
    assert len(result.gt_status) == 3
    assert np.isnan(result.metrics["absent"].ap50)
    assert result.gt_status["gt_index"].tolist() == [0, 3, 4]


def test_pr_scope_and_ties_are_compute_map_scope(tiny_dataset) -> None:
    truth, preds = tiny_dataset
    preds["confidence"] = 0.8
    curves = build_pr_observations(
        truth, preds, classes=["class_a", "class_c", "absent"], image_names=["img1"]
    )
    assert curves[0].gt_count == 1 and curves[1].gt_count == 1
    assert curves[0].confidence == [0.8, 0.8]
    assert curves[0].tp == [1, 1] and curves[0].fp == [0, 1]
    assert curves[2].ap50 is None


def test_fixed_invalid_inputs(tiny_dataset) -> None:
    truth, preds = tiny_dataset
    for kwargs in (
        {"thresholds": {"class_a": float("nan")}},
        {"iou_threshold": -1},
        {"matching_strategy": "wrong"},
        {"ap_method": "wrong"},
        {"skip_cohen_kappa": False},
    ):
        with pytest.raises(ValueError):
            evaluate_fixed(truth, preds, preds, classes=["class_a"], **{"thresholds": {}, **kwargs})
    invalid = preds.copy()
    invalid.loc[0, "bbox_x_br"] = -1
    with pytest.raises(ValueError):
        evaluate_fixed(truth, preds, invalid, classes=["class_a"], thresholds={})


def _paired_statuses():
    gt = pd.DataFrame(
        {
            "gt_index": [0, 1, 2, 3],
            "image_name": ["a", "a", "b", "c"],
            "instance_label": ["x", "x", "y", "y"],
            "detected": [True, False, False, True],
        }
    )
    new_gt = gt.assign(detected=[True, True, False, False])
    preds = gt.rename(columns={"gt_index": "pred_index", "detected": "is_tp"})
    new_preds = new_gt.rename(columns={"gt_index": "pred_index", "detected": "is_tp"})
    return gt, preds, new_gt, new_preds


def test_paired_none_is_one_pair_and_intervals_are_not_hypotheses() -> None:
    gt, preds, new_gt, new_preds = _paired_statuses()
    result = compare_paired(gt, preds, new_gt, new_preds, ["a", "b", "c"], iterations=100, seed=42)
    assert result.by_class == {} and result.adjusted.adjusted_p_values == []
    assert result.pooled.precision == bootstrap_precision_delta(
        preds, new_preds, ["a", "b", "c"], iterations=100, seed=42
    )
    expected = bootstrap_recall_delta(gt, new_gt, ["a", "b", "c"], iterations=100, seed=42)
    assert result.pooled.recall_interval.ci_lower == expected.ci_lower
    assert result.pooled.recall_interval.ci_upper == expected.ci_upper
    assert np.isnan(result.pooled.recall_interval.p_value)
    assert result.pooled.recall.p_value == 1.0 and result.pooled.recall.delta == 0


def test_explicit_family_order_and_pooled_restriction() -> None:
    gt, preds, new_gt, new_preds = _paired_statuses()
    result = compare_paired(
        gt, preds, new_gt, new_preds, ["a", "b", "c"], classes=["x"], iterations=100, seed=42
    )
    pair = result.by_class["x"]
    assert pair.precision.delta == 0.5 and pair.recall.delta == 0.5
    assert result.pooled.precision == pair.precision
    assert result.adjusted == adjust_benjamini_hochberg(
        [pair.precision.p_value, pair.recall.p_value]
    )
    assert len(result.adjusted.adjusted_p_values) == 2
    empty = compare_paired(gt, preds, new_gt, new_preds, [], classes=[])
    assert np.isnan(empty.pooled.precision.delta)


def test_statistical_frozen_arithmetic_and_failures() -> None:
    gt, preds, new_gt, new_preds = _paired_statuses()
    pair = compare_paired(
        gt, preds, new_gt, new_preds, ["a", "b", "c"], iterations=100, seed=42
    ).pooled
    assert pair.precision.delta == 0
    assert pair.precision.p_value == 1
    assert pair.precision.ci_lower == -2 / 3 and pair.precision.ci_upper == 0.4
    exact = mcnemar_recall(pd.Series([False] * 10), pd.Series([True] * 10))
    assert exact.delta == 1 and exact.p_value == 2 / 1024
    bh = adjust_benjamini_hochberg([0.01, np.nan, 0.04, 0.2])
    assert bh.adjusted_p_values == pytest.approx([0.03, np.nan, 0.06, 0.2], nan_ok=True)
    assert bh.rejected == [True, False, False, False]
    for kwargs in ({"iterations": 0}, {"q": float("nan")}, {"classes": ["x", "x"]}):
        with pytest.raises(ValueError):
            compare_paired(gt, preds, new_gt, new_preds, ["a", "b", "c"], **kwargs)


def test_explicit_rendering_single_save_caption_and_frames(tiny_dataset, tmp_path: Path) -> None:
    truth, _ = tiny_dataset
    metrics = {"class_a": Metrics(tp=3, fp=2, fn=1)}
    explicit = {
        "dashboard": tmp_path / "elsewhere/full.xlsx",
        "precision": tmp_path / "plots/p.png",
    }
    caption = "Model: literal $name$\nTraining task: example"
    real_save = Figure.savefig
    saves = []

    def observe_save(figure, path, **kwargs):
        saves.append(path)
        assert figure._suptitle.get_text() == caption
        return real_save(figure, path, **kwargs)

    with (
        patch.object(Figure, "savefig", observe_save),
        patch("digital_metrics.reporting.dashboard.plt.show") as show,
    ):
        result = render_evaluation(
            metrics,
            truth,
            None,
            [],
            "new",
            path=str(tmp_path / "new"),
            artifact_paths=explicit,
            caption=caption,
        )
        show.assert_not_called()
    assert len(saves) == 4 and len(set(saves)) == 4
    assert result.dashboard_path == explicit["dashboard"]
    assert result.confusion_matrix_path is None
    for rendered in result.plots.values():
        assert rendered.path.is_file() and rendered.caption == caption
        assert not plt.fignum_exists(rendered.figure.number)
    from PIL import Image

    with Image.open(explicit["precision"]) as image:
        assert image.info["Model identity"] == caption
    legacy, legacy_dtrk = get_dashboards(
        metrics, truth, None, [], "old", path=str(tmp_path / "old")
    )
    pd.testing.assert_frame_equal(result.dashboard, legacy)
    pd.testing.assert_frame_equal(result.dtrk_dashboard, legacy_dtrk)


def test_rendering_empty_and_explicit_destination_validation(tiny_dataset, tmp_path: Path) -> None:
    truth, _ = tiny_dataset
    empty = render_evaluation({}, truth, None, [], path=str(tmp_path), save_to_excel=False)
    assert empty.plots == {} and empty.dashboard_path is None
    with pytest.raises(ValueError, match="Unknown artifact"):
        render_evaluation({}, truth, None, [], artifact_paths={"wrong": tmp_path})


@pytest.mark.parametrize("method", ["continuous", "interp"])
@pytest.mark.parametrize("strategy", ["greedy", "iou_prior", "hungarian"])
def test_immutable_baseline_ap_arithmetic(method, strategy) -> None:
    """Values produced by the unmodified baseline, including ties/FPs/absence."""
    import json

    baseline = json.loads((Path(__file__).parent / "fixtures/shared_ap_baseline.json").read_text())
    truth = pd.DataFrame(baseline["ground_truth"])
    predictions = pd.DataFrame(baseline["predictions"])
    result = evaluate_fixed(
        truth,
        predictions,
        predictions,
        classes=["x", "y", "absent"],
        thresholds={},
        split_image_names=baseline["images"],
        matching_strategy=strategy,
        ap_method=method,
    )
    expected = baseline["results"][f"{method}:{strategy}"]
    for name, values in expected.items():
        actual = result.metrics[name]
        for value, observed in zip(values, [actual.ap50, actual.ap75, actual.ap50_95], strict=True):
            if value is None:
                assert np.isnan(observed)
            else:
                assert observed == value


def test_fixed_empty_truth_and_original_hashable_ids(tiny_dataset) -> None:
    truth, preds = tiny_dataset
    empty_truth = truth.iloc[:0]
    result = evaluate_fixed(
        empty_truth,
        preds,
        preds,
        classes=["class_a"],
        thresholds={},
        split_image_names=["img1", "img2"],
    )
    assert result.metrics["class_a"].tp == 0 and result.metrics["class_a"].fp == 5
    assert result.gt_status.empty and result.pr_curves[0].ap50 is None
    truth = truth.copy()
    preds = preds.copy()
    truth.index = [f"gt-{i}" for i in range(len(truth))]
    preds.index = [f"pred-{i}" for i in range(len(preds))]
    result = evaluate_fixed(truth, preds, preds, classes=["class_a"], thresholds={})
    assert result.gt_status["gt_index"].tolist() == ["gt-0", "gt-3", "gt-4"]
    assert result.pred_status["pred_index"].tolist() == [
        "pred-0",
        "pred-2",
        "pred-3",
        "pred-4",
        "pred-6",
    ]


def test_new_render_save_failure_releases_owned_figure(tiny_dataset, tmp_path: Path) -> None:
    truth, _ = tiny_dataset
    before = set(plt.get_fignums())
    with (
        patch.object(Figure, "savefig", side_effect=OSError("failed save")),
        pytest.raises(OSError, match="failed save"),
    ):
        render_evaluation({"class_a": Metrics(tp=1)}, truth, None, [], path=str(tmp_path))
    assert set(plt.get_fignums()) == before
