"""Tests for the grouped Evaluation config objects (additive over flat kwargs)."""

import pandas as pd
import pytest

from digital_metrics import (
    Evaluation,
    InferenceConfig,
    PreprocessConfig,
    ScoringConfig,
)

# ---------------------------------------------------------------------------
# Defaults mirror the flat-kwarg defaults
# ---------------------------------------------------------------------------


def test_config_defaults_match_flat_defaults() -> None:
    s = ScoringConfig()
    assert s.iou_threshold == 0.5
    assert s.matching_strategy == "iou_prior"
    assert s.ap_method == "interp"
    assert s.confidence_optimization == "per_class"
    assert s.skip_cohen_kappa is True

    p = PreprocessConfig()
    assert p.dedup_gt is False
    assert p.conf_threshold is None
    assert p.nms_containment_threshold is None
    assert p.nms_iou_threshold is None

    i = InferenceConfig()
    assert i.weights_path is None
    assert i.predict_kwargs is None


def test_default_construction_equivalent_to_empty_scoring_config(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    flat = Evaluation(preds_df, gt_df)
    grouped = Evaluation(preds_df, gt_df, scoring=ScoringConfig())

    assert flat.iou_threshold == grouped.iou_threshold
    assert flat.matching_strategy == grouped.matching_strategy
    assert flat._ap_method == grouped._ap_method


# ---------------------------------------------------------------------------
# Grouped config produces the same instance state as the equivalent flat kwargs
# ---------------------------------------------------------------------------


def test_scoring_config_maps_to_instance_state(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    flat = Evaluation(
        preds_df,
        gt_df,
        iou_threshold=0.3,
        matching_strategy="greedy",
        ap_method="continuous",
        confidence_optimization="global",
        skip_cohen_kappa=False,
    )
    grouped = Evaluation(
        preds_df,
        gt_df,
        scoring=ScoringConfig(
            iou_threshold=0.3,
            matching_strategy="greedy",
            ap_method="continuous",
            confidence_optimization="global",
            skip_cohen_kappa=False,
        ),
    )
    for ev in (flat, grouped):
        assert ev.iou_threshold == 0.3
        assert ev.matching_strategy == "greedy"
        assert ev._ap_method == "continuous"
        assert ev._confidence_optimization == "global"
        assert ev._skip_cohen_kappa is False


def test_preprocessing_config_applies_conf_filter(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    flat = Evaluation(preds_df, gt_df, preprocess_preds_conf_threshold=0.5)
    grouped = Evaluation(preds_df, gt_df, preprocessing=PreprocessConfig(conf_threshold=0.5))
    assert len(flat.preds_df) == len(grouped.preds_df)
    assert all(grouped.preds_df["confidence"] >= 0.5)


def test_inference_config_maps_to_instance_state(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    ev = Evaluation(
        preds_df,
        gt_df,
        inference=InferenceConfig(weights_path="best.pt", predict_kwargs={"imgsz": 1280}),
    )
    assert ev._weights_path == "best.pt"
    assert ev._predict_kwargs == {"imgsz": 1280}


# ---------------------------------------------------------------------------
# Grouped/flat produce identical evaluation results
# ---------------------------------------------------------------------------


def test_grouped_and_flat_yield_identical_metrics(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    flat = Evaluation(preds_df, gt_df, iou_threshold=0.5, matching_strategy="greedy")
    grouped = Evaluation(
        preds_df,
        gt_df,
        scoring=ScoringConfig(iou_threshold=0.5, matching_strategy="greedy"),
    )
    flat(split="all", find_best_confs=True)
    grouped(split="all", find_best_confs=True)

    assert flat.metrics.keys() == grouped.metrics.keys()
    for cls in flat.metrics:
        assert flat.metrics[cls].precision == pytest.approx(grouped.metrics[cls].precision)
        assert flat.metrics[cls].recall == pytest.approx(grouped.metrics[cls].recall)
        assert flat.best_confidences[cls] == pytest.approx(grouped.best_confidences[cls])


# ---------------------------------------------------------------------------
# Precedence: grouped config wins over the corresponding flat kwargs
# ---------------------------------------------------------------------------


def test_scoring_config_takes_precedence_over_flat(
    tiny_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = tiny_dataset
    # Flat says greedy/0.3, but the grouped config (hungarian/0.7) must win.
    ev = Evaluation(
        preds_df,
        gt_df,
        iou_threshold=0.3,
        matching_strategy="greedy",
        scoring=ScoringConfig(iou_threshold=0.7, matching_strategy="hungarian"),
    )
    assert ev.iou_threshold == 0.7
    assert ev.matching_strategy == "hungarian"
