import pandas as pd

from metrics import Evaluation
from metrics.confidence import find_best_confidences, find_best_global_confidence
from metrics.matching import match_boxes


def _tiny_matches(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    gt_df, preds_df = tiny_dataset
    classes = gt_df["instance_label"].unique().tolist()
    matches = match_boxes(
        gt_df,
        preds_df,
        iou_threshold=0.5,
        strategy="greedy",
        split_image_names=gt_df["image_name"].unique().tolist(),
    )
    return matches, classes


def test_per_class_thresholds_differ_across_classes(tiny_dataset) -> None:
    matches, classes = _tiny_matches(tiny_dataset)

    per_class = find_best_confidences(matches, classes)

    # class_a peaks at conf=0.85 (all 3 TP, no FP), class_c only at 0.60.
    assert per_class["class_a"] == 0.85
    assert per_class["class_c"] == 0.60
    assert len(set(per_class.values())) > 1  # genuinely per-class


def test_global_confidence_is_single_shared_threshold(tiny_dataset) -> None:
    matches, classes = _tiny_matches(tiny_dataset)

    threshold = find_best_global_confidence(matches, classes)

    # Mean per-class F1 is maximised at 0.60: class_a is perfect (P=R=1),
    # class_b and class_c still keep their single TP.
    assert threshold == 0.60


def test_global_confidence_empty_matches_returns_zero() -> None:
    assert find_best_global_confidence({}, ["class_a"]) == 0.0


def test_evaluation_global_mode_applies_one_threshold(tiny_dataset) -> None:
    gt_df, preds_df = tiny_dataset

    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5, confidence_optimization="global")
    ev(split="test", find_best_confs=True)

    confidences = set(ev.best_confidences.values())
    assert len(confidences) == 1  # YOLO-style: one threshold for every class
    assert confidences == {0.60}


def test_evaluation_per_class_mode_is_default(tiny_dataset) -> None:
    gt_df, preds_df = tiny_dataset

    ev = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    ev(split="test", find_best_confs=True)

    assert len(set(ev.best_confidences.values())) > 1
