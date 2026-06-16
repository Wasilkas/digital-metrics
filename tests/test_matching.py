import pandas as pd
import pytest

from metrics.matching import match_boxes


def _count(matches: dict[str, list[object]], label: str, mtype: str) -> int:  # type: ignore[type-arg]
    return sum(1 for m in matches.get(label, []) if m.type == mtype)


def test_greedy_returns_all_classes(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    assert set(matches.keys()) >= {"class_a", "class_b", "class_c"}


def test_greedy_tp_fp_fn_class_a(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    assert _count(matches, "class_a", "TP") == 3
    assert _count(matches, "class_a", "FP") == 2
    assert _count(matches, "class_a", "FN") == 0


def test_greedy_tp_fp_fn_class_b(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    assert _count(matches, "class_b", "TP") == 1
    assert _count(matches, "class_b", "FP") == 0
    assert _count(matches, "class_b", "FN") == 1


def test_greedy_tp_fp_fn_class_c(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    assert _count(matches, "class_c", "TP") == 1
    assert _count(matches, "class_c", "FP") == 0
    assert _count(matches, "class_c", "FN") == 1


def test_hungarian_returns_same_keys(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    greedy = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    hungarian = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="hungarian")
    assert set(hungarian.keys()) == set(greedy.keys())


def test_hungarian_tp_ge_greedy(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    greedy = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    hungarian = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="hungarian")

    total_tp_greedy = sum(_count(greedy, c, "TP") for c in ("class_a", "class_b", "class_c"))
    total_tp_hungarian = sum(_count(hungarian, c, "TP") for c in ("class_a", "class_b", "class_c"))
    # Hungarian is globally optimal: total TP must be ≥ greedy
    assert total_tp_hungarian >= total_tp_greedy


def test_greedy_bug_fix_no_double_gt_claim() -> None:
    """A GT consumed by a label-mismatch FP must not also appear as FN."""
    gt_df = pd.DataFrame(
        [("img", "class_a", 0, 0, 100, 100, "test")],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "split",
        ],
    )
    # Two preds competing for the same GT; first one has wrong label
    preds_df = pd.DataFrame(
        [
            ("img", "class_b", 0, 0, 100, 100, 0.9),  # label mismatch → FP, GT consumed
            ("img", "class_a", 0, 0, 100, 100, 0.5),  # should NOT get TP (GT consumed)
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "confidence",
        ],
    )
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    # class_a: one pred with IoU=1.0 but GT already consumed → FP
    assert _count(matches, "class_a", "TP") == 0
    # class_b: the label-mismatch pred is an FP
    assert _count(matches, "class_b", "FP") == 1
    # GT is consumed by class_b pred, so NOT a FN for class_a
    assert _count(matches, "class_a", "FN") == 0


def test_iou_prior_returns_same_keys(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    greedy = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    iou_prior = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="iou_prior")
    assert set(iou_prior.keys()) == set(greedy.keys())


def test_iou_prior_tp_fp_fn(tiny_dataset: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    gt_df, preds_df = tiny_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="iou_prior")
    assert _count(matches, "class_a", "TP") == 3
    assert _count(matches, "class_a", "FP") == 2
    assert _count(matches, "class_a", "FN") == 0
    assert _count(matches, "class_b", "TP") == 1
    assert _count(matches, "class_b", "FN") == 1
    assert _count(matches, "class_c", "TP") == 1
    assert _count(matches, "class_c", "FN") == 1


@pytest.fixture
def iou_prior_vs_greedy_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One GT; two preds of the same class competing for it.

    pred_a (conf=0.9) has lower IoU with the GT than pred_b (conf=0.5).

    Greedy assigns pred_a (higher confidence) → pred_b is FP.
    IoU-prior assigns pred_b (higher IoU) → pred_a is FP.
    """
    gt_df = pd.DataFrame(
        [("img", "cls", 0, 0, 100, 100, "test")],
        columns=["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl",
                 "bbox_x_br", "bbox_y_br", "split"],
    )
    preds_df = pd.DataFrame(
        [
            ("img", "cls", 0, 0, 80, 100, 0.9),   # IoU = 0.80, high conf
            ("img", "cls", 0, 0, 100, 100, 0.5),  # IoU = 1.00, low conf
        ],
        columns=["image_name", "instance_label", "bbox_x_tl", "bbox_y_tl",
                 "bbox_x_br", "bbox_y_br", "confidence"],
    )
    return gt_df, preds_df


def test_iou_prior_picks_highest_iou(
    iou_prior_vs_greedy_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = iou_prior_vs_greedy_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="iou_prior")
    # The low-confidence pred (IoU=1.0) should win the GT
    tp_matches = [m for m in matches.get("cls", []) if m.type == "TP"]
    assert len(tp_matches) == 1
    assert tp_matches[0].confidence == pytest.approx(0.5)


def test_greedy_picks_highest_confidence(
    iou_prior_vs_greedy_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = iou_prior_vs_greedy_dataset
    matches = match_boxes(gt_df, preds_df, iou_threshold=0.5, strategy="greedy")
    # The high-confidence pred (IoU=0.8) should win the GT
    tp_matches = [m for m in matches.get("cls", []) if m.type == "TP"]
    assert len(tp_matches) == 1
    assert tp_matches[0].confidence == pytest.approx(0.9)
