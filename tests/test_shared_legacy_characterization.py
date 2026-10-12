"""Legacy API observations captured before extraction, 2026-10-12."""

import inspect
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from digital_metrics import Evaluation, Metrics
from digital_metrics.reporting import get_dashboards, plot_confidence_intervals
from digital_metrics.scoring import compute_map


def test_legacy_public_defaults() -> None:
    signature = inspect.signature(compute_map)
    assert list(signature.parameters) == [
        "gt_df",
        "preds_df",
        "metrics",
        "split_image_names",
        "method",
        "strategy",
    ]
    assert signature.parameters["method"].default == "continuous"
    assert signature.parameters["strategy"].default == "greedy"
    assert inspect.signature(Evaluation).parameters["ap_method"].default == "interp"
    assert Metrics().cohen_kappa == 0


@pytest.mark.parametrize("strategy", ["greedy", "iou_prior", "hungarian"])
@pytest.mark.parametrize("method", ["continuous", "interp"])
def test_legacy_ap_mutation_and_absent_class(tiny_dataset, strategy, method) -> None:
    gt, preds = tiny_dataset
    original_gt, original_preds = gt.copy(), preds.copy()
    metric = Metrics(tp=3)
    metrics = {"class_a": metric, "absent": Metrics()}
    assert compute_map(gt, preds, metrics, method=method, strategy=strategy) is None
    assert metrics["class_a"] is metric
    assert metric.tp == 3
    assert metric.ap50 == pytest.approx(1 if method == "continuous" else 0.9980000001192093)
    assert np.isnan(metrics["absent"].ap50)
    pd.testing.assert_frame_equal(gt, original_gt)
    pd.testing.assert_frame_equal(preds, original_preds)


def test_legacy_artifact_names_save_and_show(tiny_dataset, tmp_path: Path) -> None:
    gt, _ = tiny_dataset
    metrics = {"class_a": Metrics(tp=1)}
    result = get_dashboards(
        metrics,
        gt,
        np.zeros((2, 2), dtype=np.int64),
        ["class_a", "background"],
        suffix="observed",
        path=str(tmp_path),
    )
    assert isinstance(result, tuple) and len(result) == 2
    assert {p.name for p in tmp_path.iterdir()} == {
        "full_dashboard_observed.xlsx",
        "метрики_дтрк_observed.xlsx",
        "matrix_observed.xlsx",
        *[
            f"{m}_confidence_intervals_observed.png"
            for m in ("recall", "precision", "perebrak", "nedobrak")
        ],
    }
    with patch("digital_metrics.reporting.dashboard.plt.show") as show:
        figure, _ = plot_confidence_intervals(metrics, "precision")
        show.assert_called_once()
    import matplotlib.pyplot as plt

    plt.close(figure)
    with (
        patch("digital_metrics.reporting.dashboard.plt.savefig") as save,
        patch("digital_metrics.reporting.dashboard.plt.show") as show,
    ):
        plot_confidence_intervals(metrics, "precision", save_path=str(tmp_path / "x.png"))
        save.assert_called_once()
        show.assert_not_called()
