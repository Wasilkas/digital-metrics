"""Direct tests for the scoring engines and Evaluation's engine selection."""

import math

import pandas as pd
import pytest

from metrics import Evaluation
from metrics.calibration import ConfidenceCalibrator
from metrics.engines import BackendEngine, EvaluationResult, NativeEngine, ScoringInputs


def _native_engine(classes: list[str]) -> NativeEngine:
    return NativeEngine(
        classes=classes,
        iou_threshold=0.5,
        matching_strategy="greedy",
        ap_method="interp",
        skip_cohen_kappa=True,
        calibrator=ConfidenceCalibrator(
            classes=classes,
            iou_threshold=0.5,
            matching_strategy="greedy",
            confidence_optimization="per_class",
        ),
    )


def _inputs(gt_df: pd.DataFrame, preds_df: pd.DataFrame, *, split: str = "all") -> ScoringInputs:
    eval_gt = gt_df if split == "all" else gt_df[gt_df["split"] == split]
    return ScoringInputs(
        gt_df=eval_gt,
        preds_df=preds_df,
        raw_preds_df=preds_df,
        split_df=gt_df,
        split=split,
        find_best_confs=True,
        calibration_split=None,
    )


# ---------------------------------------------------------------------------
# NativeEngine
# ---------------------------------------------------------------------------


def test_native_engine_run_produces_full_result(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    classes = ["class_a", "class_b", "class_c"]
    result = _native_engine(classes).run(_inputs(gt_df, preds_df))

    assert isinstance(result, EvaluationResult)
    assert set(result.metrics) == set(classes)
    assert result.detection_metrics == {}  # native produces none
    assert result.matches and result.unfiltered_matches  # native fills matches
    assert result.cm is not None
    assert result.class_labels[-1] == "background"
    for m in result.metrics.values():
        assert m.ap50 >= 0.0


def test_native_engine_resolve_calibration_split_is_identity() -> None:
    engine = _native_engine(["a"])
    assert engine.resolve_calibration_split("val") == "val"
    assert engine.resolve_calibration_split(None) is None


def test_native_engine_matches_evaluation_metrics(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The engine run reproduces what Evaluation reports end-to-end."""
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(
        preds_df, gt_df, iou_threshold=0.5, matching_strategy="greedy", ap_method="interp"
    )
    ev(split="all", find_best_confs=True)

    result = _native_engine(["class_a", "class_b", "class_c"]).run(_inputs(gt_df, preds_df))
    for cls in ev.metrics:
        assert result.metrics[cls].precision == pytest.approx(ev.metrics[cls].precision)
        assert result.metrics[cls].recall == pytest.approx(ev.metrics[cls].recall)
        if not math.isnan(ev.metrics[cls].ap50):
            assert result.metrics[cls].ap50 == pytest.approx(ev.metrics[cls].ap50)


# ---------------------------------------------------------------------------
# BackendEngine (torch-free bits only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["ultralytics", "torchmetrics"])
def test_backend_engine_resolve_calibration_split_keeps_split(backend: str) -> None:
    # Both backends now honour a calibration split (no decline / no-op).
    engine = BackendEngine(
        backend=backend,  # type: ignore[arg-type]
        classes=["a"],
        confidence_optimization="per_class",
        calibrator=ConfidenceCalibrator(
            classes=["a"],
            iou_threshold=0.5,
            matching_strategy="greedy",
            confidence_optimization="per_class",
        ),
    )
    assert engine.resolve_calibration_split("val") == "val"
    assert engine.resolve_calibration_split(None) is None


# ---------------------------------------------------------------------------
# Evaluation engine selection
# ---------------------------------------------------------------------------


def test_evaluation_selects_native_engine_by_default(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df)
    assert isinstance(ev._engine, NativeEngine)


def test_evaluation_selects_backend_engine_when_backend_set(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(preds_df, gt_df, backend="ultralytics")
    assert isinstance(ev._engine, BackendEngine)
