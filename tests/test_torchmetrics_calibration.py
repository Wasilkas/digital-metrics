"""Tests for torchmetrics-backend calibration.

The curve-reading helpers are torch-free (they operate on numpy arrays shaped
like pycocotools' precision/score tensors), so they run without the extra. The
end-to-end Evaluation calibration test needs the ``torchmetrics`` extra and is
skipped otherwise.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from metrics import Evaluation
from metrics.backends.torchmetrics_metrics import (
    _conf_at_max_f1,
    _global_conf_at_max_mean_f1,
    _read_prf1_at_conf,
)


def _curve(points: dict[int, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Build (precision, score) length-101 curves; -1 everywhere except `points`.

    `points` maps a recall-threshold index (recall = idx/100) to (precision, score).
    """
    prec = np.full(101, -1.0)
    score = np.full(101, -1.0)
    for idx, (p, s) in points.items():
        prec[idx] = p
        score[idx] = s
    return prec, score


# ---------------------------------------------------------------------------
# _read_prf1_at_conf
# ---------------------------------------------------------------------------


def test_read_prf1_at_conf_picks_max_recall_above_threshold() -> None:
    # Two reachable points: recall 0.3 @ score 0.8, recall 0.6 @ score 0.4.
    prec, score = _curve({30: (0.5, 0.8), 60: (0.9, 0.4)})

    # conf 0.4 keeps both → max-recall point (0.6, prec 0.9).
    p, r, f1 = _read_prf1_at_conf(prec, score, 0.4)
    assert (p, r) == pytest.approx((0.9, 0.6))
    assert f1 == pytest.approx(2 * 0.9 * 0.6 / (0.9 + 0.6))

    # conf 0.5 drops the 0.4-score point → only recall 0.3 (prec 0.5).
    p, r, f1 = _read_prf1_at_conf(prec, score, 0.5)
    assert (p, r) == pytest.approx((0.5, 0.3))


def test_read_prf1_at_conf_above_all_scores_is_zero() -> None:
    prec, score = _curve({30: (0.5, 0.8), 60: (0.9, 0.4)})
    assert _read_prf1_at_conf(prec, score, 0.95) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# _conf_at_max_f1
# ---------------------------------------------------------------------------


def test_conf_at_max_f1_returns_score_at_best_f1_point() -> None:
    # f1(0.3, 0.5) = 0.375 < f1(0.6, 0.9) = 0.72 → best is the second point (score 0.4).
    prec, score = _curve({30: (0.5, 0.8), 60: (0.9, 0.4)})
    assert _conf_at_max_f1(prec, score) == pytest.approx(0.4)


def test_conf_at_max_f1_no_reachable_point_is_zero() -> None:
    prec = np.full(101, -1.0)
    score = np.full(101, -1.0)
    assert _conf_at_max_f1(prec, score) == 0.0


# ---------------------------------------------------------------------------
# _global_conf_at_max_mean_f1
# ---------------------------------------------------------------------------


def test_global_conf_maximises_mean_f1_across_classes() -> None:
    # Shared scores 0.8 / 0.4 across two classes; mean F1 is higher at 0.4.
    c1 = _curve({30: (0.5, 0.8), 60: (0.9, 0.4)})
    c2 = _curve({30: (0.9, 0.8), 60: (0.5, 0.4)})
    assert _global_conf_at_max_mean_f1([c1, c2]) == pytest.approx(0.4)


def test_global_conf_empty_is_zero() -> None:
    assert _global_conf_at_max_mean_f1([]) == 0.0


# ---------------------------------------------------------------------------
# End-to-end (needs the torchmetrics extra)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("torchmetrics") is None,
    reason="torchmetrics not installed",
)
def test_torchmetrics_backend_calibration_when_installed(
    split_dataset: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    gt_df, preds_df = split_dataset
    base = Evaluation(preds_df, gt_df, backend="torchmetrics")
    base(split="test")  # self-selected operating point

    cal = Evaluation(preds_df, gt_df, backend="torchmetrics")
    cal(split="test", calibration_split="val")  # operating point from val

    assert cal.detection_metrics  # populated
    assert cal.cm is None  # torchmetrics has no confusion matrix
    assert set(cal.best_confidences) >= set(cal.detection_metrics)  # thresholds recorded
    # AP is read over the full curve, so calibration must not change it.
    for cls, dm in base.detection_metrics.items():
        assert cal.detection_metrics[cls].ap50 == pytest.approx(dm.ap50)
        assert cal.detection_metrics[cls].ap50_95 == pytest.approx(dm.ap50_95)
