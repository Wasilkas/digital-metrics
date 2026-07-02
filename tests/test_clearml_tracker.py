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
    """Models a manually built (``Task.create``-style, non-main) task.

    It starts in the "created"/draft status and, like the real non-main task,
    ``close()`` does not complete it — only the tracker's explicit status handling
    does. ``is_main`` can be flipped to model a ``Task.init`` main task.
    """

    def __init__(self, *, is_main: bool = False) -> None:
        self._logger = _FakeLogger()
        self.artifacts: dict[str, Any] = {}
        self.closed = False
        self.status = "created"
        self._is_main = is_main

    def get_logger(self) -> _FakeLogger:
        return self._logger

    def upload_artifact(self, name: str, artifact_object: Any) -> None:
        self.artifacts[name] = artifact_object

    def is_main_task(self) -> bool:
        return self._is_main

    def mark_started(self) -> None:
        self.status = "in_progress"

    def mark_completed(self, **_: Any) -> None:
        self.status = "completed"

    def close(self) -> None:
        # A real non-main task's status is unchanged by close(); a main task
        # completes. Mirror that so tests catch missing explicit status handling.
        if self._is_main and self.status not in {"completed", "failed", "stopped"}:
            self.status = "completed"
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


def test_injected_draft_task_status_started_then_completed() -> None:
    # A Task.create-style task arrives in "created"/draft: the tracker must start it
    # and complete it on close (Bug: it stayed draft because close() ignores non-main).
    task = _FakeTask()
    assert task.status == "created"

    tracker = ClearMLTracker(task=task, attach_logs=False)
    assert task.status == "in_progress"  # started on construction

    tracker.close()
    assert task.status == "completed"  # explicitly completed for a non-main task
    assert task.closed


def test_injected_main_task_completed_by_close_not_double_marked() -> None:
    # A Task.init main task is already running and is completed by close() itself;
    # the tracker must not try to mark_completed a main task.
    task = _FakeTask(is_main=True)
    task.status = "in_progress"

    tracker = ClearMLTracker(task=task, attach_logs=False)
    assert task.status == "in_progress"  # not re-started

    tracker.close()
    assert task.status == "completed"
    assert task.closed


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
