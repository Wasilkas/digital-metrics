"""Torch/ClearML-free tests for the ClearML tracking layer.

An injected fake ``Task`` (see ``ClearMLTracker(task=...)``) lets us exercise the
whole dispatch — scalars, artifacts, plots and the loguru sink — without the
optional ``clearml`` extra installed.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from digital_metrics import Evaluation, Metrics
from digital_metrics.tracking import ClearMLTracker, summarize_metrics


class _FakeLogger:
    def __init__(self) -> None:
        self.scalars: list[dict[str, Any]] = []
        self.tables: list[dict[str, Any]] = []
        self.single_values: dict[str, float] = {}
        self.images: list[dict[str, Any]] = []
        self.confusion_matrices: list[dict[str, Any]] = []
        self.texts: list[str] = []

    def report_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.scalars.append({"title": title, "series": series, "value": value})

    def report_table(self, title: str, series: str, iteration: int, table_plot: Any) -> None:
        self.tables.append({"title": title, "rows": len(table_plot)})

    def report_single_value(self, name: str, value: float) -> None:
        self.single_values[name] = value

    def report_image(self, title: str, series: str, iteration: int, local_path: str) -> None:
        self.images.append({"series": series, "local_path": local_path})

    def report_confusion_matrix(self, **kwargs: Any) -> None:
        self.confusion_matrices.append(kwargs)

    def report_text(self, text: str, *, print_console: bool = True) -> None:
        self.texts.append(text)


class _FakeTask:
    def __init__(self) -> None:
        self._logger = _FakeLogger()
        self.artifacts: dict[str, Any] = {}
        self.closed = False

    def get_logger(self) -> _FakeLogger:
        return self._logger

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        self.artifacts[name] = artifact_object

    def close(self) -> None:
        self.closed = True


def _metrics(**ap: float) -> Metrics:
    return Metrics(tp=8, fp=2, fn=2, confidence=0.5, **ap)


def test_summarize_metrics_nanmean_excludes_absent_class() -> None:
    metrics = {
        "a": _metrics(ap50=0.8, ap75=0.6, ap50_95=0.5),
        "b": _metrics(ap50=float("nan"), ap75=float("nan"), ap50_95=float("nan")),
    }
    df, means = summarize_metrics(metrics)

    assert list(df.index) == ["a", "b"]
    # AP means ignore the NaN class; precision (computed field) averages both.
    assert means["mean_ap50"] == 0.8
    assert means["mean_precision"] == df["precision"].mean()


def test_log_evaluation_dispatches_all_layers(tiny_dataset, tmp_path) -> None:
    gt_df, preds_df = tiny_dataset
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5, matching_strategy="greedy")
    evaluation("test")

    task = _FakeTask()
    tracker = ClearMLTracker(task=task, attach_logs=False)
    devs, dtrk = tracker.log_evaluation(evaluation, artifacts_dir=str(tmp_path))

    log = task.get_logger()
    # Scalars: at least one per-class point per core metric + the table + means.
    assert {s["title"] for s in log.scalars} >= {"precision", "recall", "f1_score"}
    assert log.tables and log.tables[0]["rows"] == len(devs)
    assert "mean_precision" in log.single_values

    # Artifacts: dashboards, thresholds, confusion matrix + written Excel files.
    assert {"dashboard_full", "dashboard_dtrk", "best_confidences", "confusion_matrix"} <= set(
        task.artifacts
    )
    assert any(name.endswith(".xlsx") for name in task.artifacts)

    # Plots: four CI images + one confusion-matrix plot.
    assert {img["series"] for img in log.images} == {
        "recall",
        "precision",
        "perebrak",
        "nedobrak",
    }
    assert len(log.confusion_matrices) == 1
    assert not dtrk.empty


def test_loguru_sink_attaches_and_detaches() -> None:
    task = _FakeTask()
    tracker = ClearMLTracker(task=task, attach_logs=True)

    logger.info("hello from evaluation")
    assert any("hello from evaluation" in text for text in task.get_logger().texts)

    sink_id = tracker._sink_id
    assert sink_id is not None
    tracker.close()
    assert tracker._sink_id is None
    assert task.closed
