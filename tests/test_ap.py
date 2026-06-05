import numpy as np
import pandas as pd
import pytest

from metrics.ap import compute_ap, compute_map
from metrics.types import Metrics


def test_ap_perfect_detector() -> None:
    # Recall rises from 0→1 with precision held at 1 → AP = 1.0
    recall = np.array([0.1, 0.2, 0.5, 1.0])
    precision = np.array([1.0, 1.0, 1.0, 1.0])
    assert compute_ap(recall, precision) == pytest.approx(1.0)


def test_ap_zero_precision() -> None:
    # All predictions are wrong; precision = 0 everywhere → AP = 0
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([0.0, 0.0, 0.0])
    assert compute_ap(recall, precision) == pytest.approx(0.0)


def test_ap_in_unit_interval() -> None:
    rng = np.random.default_rng(42)
    recall = np.sort(rng.uniform(0, 1, 20))
    precision = rng.uniform(0, 1, 20)
    ap = compute_ap(recall, precision)
    assert 0.0 <= ap <= 1.0


def _make_perfect_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfect detector: one class, preds exactly match GTs."""
    cols_gt = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "split",
    ]
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "test"), ("img2", "cat", 0, 0, 100, 100, "test")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
    ]
    preds = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, 0.99), ("img2", "cat", 0, 0, 100, 100, 0.95)],
        columns=cols_p,
    )
    return gt, preds


def test_compute_map_perfect() -> None:
    gt, preds = _make_perfect_dataset()
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics)
    assert metrics["cat"].ap50 == pytest.approx(1.0, abs=1e-6)
    assert metrics["cat"].ap75 == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= metrics["cat"].ap50_95 <= 1.0


def test_compute_map_all_fp() -> None:
    """All predictions are FP (wrong location) → mAP ≈ 0."""
    cols_gt = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "split",
    ]
    gt = pd.DataFrame(
        [("img1", "cat", 0, 0, 100, 100, "test")],
        columns=cols_gt,
    )
    cols_p = [
        "image_name",
        "instance_label",
        "bbox_x_tl",
        "bbox_y_tl",
        "bbox_x_br",
        "bbox_y_br",
        "confidence",
    ]
    preds = pd.DataFrame(
        [("img1", "cat", 500, 500, 600, 600, 0.9)],  # no overlap with GT
        columns=cols_p,
    )
    metrics: dict[str, Metrics] = {"cat": Metrics()}
    compute_map(gt, preds, metrics)
    assert metrics["cat"].ap50 == pytest.approx(0.0, abs=1e-6)
