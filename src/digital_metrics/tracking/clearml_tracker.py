"""ClearML experiment tracking as a separate, non-intrusive layer.

The tracker *consumes* an already-run :class:`~digital_metrics.evaluation.Evaluation`
(its ``metrics`` / ``cm`` / ``best_confidences`` and the dashboards/plots it
produces) and mirrors them into a ClearML ``Task``. The core evaluation code
knows nothing about ClearML — this layer sits on top and can be dropped in or
left out entirely.

``clearml`` is a heavy optional dependency, so it is imported lazily and kept out
of the core install. Enable it with::

    pip install digital-metrics[clearml]

Four things are logged (all opt-out-able):

* **Scalars** — per-class P/R/F1/mAP as scalar plots + a per-class metrics table,
  and the headline means (nan-aware for AP) as single values.
* **Artifacts** — the analyst/production dashboard DataFrames, the written Excel
  files, ``best_confidences`` and the confusion matrix.
* **Plots** — the confidence-interval PNGs as images and the confusion matrix as
  a ClearML confusion-matrix plot.
* **Logs** — an optional ``loguru`` sink forwarding run logs to the ClearML
  console.
"""

from __future__ import annotations

import os
from types import TracebackType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from loguru import logger

if TYPE_CHECKING:
    from clearml import Logger, Task

    from ..evaluation import Evaluation

# Core per-class metrics mirrored as ClearML scalar plots (one plot per metric,
# one series per class). All are columns of the per-class metrics DataFrame.
CORE_METRICS = ("precision", "recall", "f1_score", "ap50", "ap75", "ap50_95")

# Metrics whose class average must ignore NaN (AP is NaN for classes absent from
# the evaluated split — see the mAP notes in CLAUDE.md).
_NAN_AWARE = ("ap50", "ap75", "ap50_95")

# The confidence-interval PNGs get_dashboards writes into its output directory.
_CI_PLOT_METRICS = ("recall", "precision", "perebrak", "nedobrak")

# ClearML task statuses (clearml.Task.TaskStatusEnum). A task created via
# ``Task.create`` starts as "created" (shown as "draft" in the UI) and only leaves
# it once explicitly started; these two sets drive the status management below.
_DRAFT_STATUSES = frozenset({"created", "draft"})
_FINISHED_STATUSES = frozenset({"completed", "published", "closed", "failed", "stopped"})


def _import_task() -> type[Task]:
    """Import ``clearml.Task`` lazily with an install hint on failure."""
    try:
        from clearml import Task
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ClearMLTracker requires the optional 'clearml' dependency. "
            "Install it with: pip install digital-metrics[clearml]"
        ) from exc
    return cast("type[Task]", Task)


def summarize_metrics(metrics: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, float]]:
    """Flatten per-class ``Metrics`` to a DataFrame and headline means.

    Args:
        metrics: Per-class ``Metrics`` objects (``Evaluation.metrics``).

    Returns:
        ``(df, means)`` — the per-class metrics DataFrame (index = class name) and
        a ``{"mean_<metric>": value}`` dict over :data:`CORE_METRICS`. Means for
        AP metrics use ``nanmean`` so classes absent from the split are excluded.

    This is torch/clearml-free so it can be unit-tested without the extra.
    """
    df = pd.DataFrame.from_dict({k: v.model_dump() for k, v in metrics.items()}, orient="index")
    means: dict[str, float] = {}
    for metric in CORE_METRICS:
        if metric not in df.columns:
            continue
        values = df[metric].to_numpy(dtype=float)
        reduce = np.nanmean if metric in _NAN_AWARE else np.mean
        means[f"mean_{metric}"] = float(reduce(values)) if len(values) else float("nan")
    return df, means


class ClearMLTracker:
    """Mirror an :class:`Evaluation`'s results into a ClearML ``Task``.

    Either pass an existing ``task`` (created however you like) or let the tracker
    call ``Task.init`` from ``project_name`` / ``task_name`` and any extra
    ``Task.init`` keyword arguments. Usable as a context manager, which closes the
    task (and detaches the loguru sink) on exit::

        with ClearMLTracker(project_name="detector", task_name="run-42") as tracker:
            evaluation("test", calibration_split="val")
            tracker.log_evaluation(evaluation)

    Task lifecycle is handled so both entry points behave as expected:

    * **``project_name`` / ``task_name``** — ClearML allows only one *main* task per
      process, so a second ``Task.init`` with a different name would otherwise raise
      ("task name does not match"). The tracker closes any still-open current task
      first, so creating several trackers in one process just works.
    * **injected ``task``** — a manually built task (e.g. ``Task.create``) starts in
      the "created"/draft status and is *not* completed by ``Task.close`` (that only
      finalises the main task). The tracker moves it to running on construction and
      to completed on :meth:`close`, so an injected task's status tracks the run.
    """

    def __init__(
        self,
        task: Task | None = None,
        *,
        project_name: str | None = None,
        task_name: str | None = None,
        output_uri: str | bool | None = None,
        attach_logs: bool = True,
        log_level: str = "INFO",
        **task_init_kwargs: Any,
    ) -> None:
        """Create the tracker.

        Args:
            task: An existing ClearML ``Task`` to log into. When ``None`` a new
                task is created via ``Task.init``.
            project_name: ClearML project (used only when ``task`` is ``None``).
            task_name: ClearML task name (used only when ``task`` is ``None``).
            output_uri: Passed to ``Task.init`` to set where artifacts/models are
                uploaded (used only when ``task`` is ``None``).
            attach_logs: Attach a ``loguru`` sink forwarding run logs to the
                ClearML console immediately. Defaults to ``True``.
            log_level: Minimum level for the loguru sink.
            **task_init_kwargs: Extra keyword arguments forwarded to ``Task.init``.
        """
        if task is not None:
            self.task: Task = task
        else:
            # Import ClearML only when we actually create a task; an injected task
            # (e.g. in tests) needs no extra installed.
            task_cls = _import_task()
            # Only one main task may exist per process: close any still-open current
            # task so a second Task.init (e.g. a new task_name) does not raise.
            current: Task | None = task_cls.current_task()
            if current is not None:
                logger.info("Closing the current ClearML task before starting a new one.")
                current.close()
            self.task = task_cls.init(
                project_name=project_name,
                task_name=task_name,
                output_uri=output_uri,
                **task_init_kwargs,
            )
        # A freshly created (draft) task must be started so its status reflects the
        # run; a Task.init task is already running, so this is a no-op there.
        self._start_if_draft()
        self._logger: Logger = self.task.get_logger()
        self._sink_id: int | None = None
        if attach_logs:
            self.attach_loguru(level=log_level)

    def _start_if_draft(self) -> None:
        """Move a task still in the 'created'/draft status to running."""
        if getattr(self.task, "status", None) in _DRAFT_STATUSES:
            self.task.mark_started()

    # -- logging entry point --------------------------------------------------

    def log_evaluation(
        self,
        evaluation: Evaluation,
        *,
        iteration: int = 0,
        artifacts_dir: str = "metrics/",
        save_to_excel: bool = True,
        save_confusion_matrix: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Log scalars, artifacts and plots for a completed evaluation.

        Runs ``evaluation.get_dashboards`` once (which writes the Excel files and
        the confidence-interval PNGs into ``artifacts_dir``) and mirrors
        everything into the ClearML task.

        Args:
            evaluation: A run ``Evaluation`` (call it before logging).
            iteration: ClearML iteration for scalar/plot reports.
            artifacts_dir: Directory the dashboards/plots are written to.
            save_to_excel: Forwarded to ``get_dashboards``.
            save_confusion_matrix: Forwarded to ``get_dashboards``.

        Returns:
            ``(devs, dtrk)`` — the dashboards from ``get_dashboards``.
        """
        assert evaluation.metrics, "Call evaluation() before log_evaluation()."
        os.makedirs(artifacts_dir, exist_ok=True)

        devs, dtrk = evaluation.get_dashboards(
            save_to_excel=save_to_excel,
            path=artifacts_dir,
            save_confusion_matrix=save_confusion_matrix,
        )
        self.log_scalars(evaluation.metrics, iteration=iteration)
        self.log_artifacts(evaluation, devs, dtrk, artifacts_dir=artifacts_dir)
        self.log_plots(evaluation, artifacts_dir=artifacts_dir, iteration=iteration)
        return devs, dtrk

    # -- individual layers ----------------------------------------------------

    def log_scalars(self, metrics: dict[str, Any], *, iteration: int = 0) -> dict[str, float]:
        """Report per-class scalars, a per-class table, and headline means.

        Returns the ``{"mean_<metric>": value}`` dict for convenience.
        """
        df, means = summarize_metrics(metrics)

        for metric in CORE_METRICS:
            if metric not in df.columns:
                continue
            for class_name, value in df[metric].items():
                if pd.isna(value):
                    continue
                self._logger.report_scalar(
                    title=metric,
                    series=str(class_name),
                    value=float(value),
                    iteration=iteration,
                )

        self._logger.report_table(
            title="metrics",
            series="per_class",
            iteration=iteration,
            table_plot=df,
        )
        for name, value in means.items():
            self._logger.report_single_value(name, value)

        return means

    def log_artifacts(
        self,
        evaluation: Evaluation,
        devs: pd.DataFrame,
        dtrk: pd.DataFrame,
        *,
        artifacts_dir: str = "metrics/",
    ) -> None:
        """Upload dashboards, thresholds, confusion matrix and the Excel files."""
        self.task.upload_artifact("dashboard_full", devs)
        self.task.upload_artifact("dashboard_dtrk", dtrk)
        self.task.upload_artifact("best_confidences", dict(evaluation.best_confidences))

        if evaluation.cm is not None and evaluation.class_labels:
            cm_df = pd.DataFrame(
                evaluation.cm,
                index=evaluation.class_labels,
                columns=evaluation.class_labels,
            )
            self.task.upload_artifact("confusion_matrix", cm_df)

        # The exact .xlsx files get_dashboards wrote, uploaded verbatim.
        suffix = evaluation.suffix
        for filename in (
            f"full_dashboard_{suffix}.xlsx",
            f"метрики_дтрк_{suffix}.xlsx",
            f"matrix_{suffix}.xlsx",
        ):
            file_path = os.path.join(artifacts_dir, filename)
            if os.path.exists(file_path):
                self.task.upload_artifact(filename, file_path)

    def log_plots(
        self,
        evaluation: Evaluation,
        *,
        artifacts_dir: str = "metrics/",
        iteration: int = 0,
    ) -> None:
        """Report the CI PNGs as images and the confusion matrix as a CM plot."""
        for metric in _CI_PLOT_METRICS:
            plot_path = os.path.join(artifacts_dir, f"{metric}_confidence_intervals.png")
            if os.path.exists(plot_path):
                self._logger.report_image(
                    title="confidence_intervals",
                    series=metric,
                    iteration=iteration,
                    local_path=plot_path,
                )

        if evaluation.cm is not None and evaluation.class_labels:
            self._logger.report_confusion_matrix(
                title="confusion_matrix",
                series="eval",
                matrix=np.asarray(evaluation.cm),
                iteration=iteration,
                xlabels=evaluation.class_labels,
                ylabels=evaluation.class_labels,
                xaxis="Predicted",
                yaxis="Ground truth",
            )

    # -- loguru bridge --------------------------------------------------------

    def attach_loguru(self, *, level: str = "INFO") -> int:
        """Route ``loguru`` records to the ClearML console. Idempotent."""
        if self._sink_id is not None:
            return self._sink_id

        def sink(message: Any) -> None:
            self._logger.report_text(str(message).rstrip("\n"), print_console=False)

        self._sink_id = logger.add(sink, level=level)
        return self._sink_id

    def detach_loguru(self) -> None:
        """Remove the loguru sink installed by :meth:`attach_loguru`."""
        if self._sink_id is not None:
            logger.remove(self._sink_id)
            self._sink_id = None

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        """Detach the loguru sink and close the ClearML task.

        ``Task.close`` only finalises the *main* task (the one ``Task.init``
        created); a non-main task — e.g. an injected ``Task.create`` one — is left
        in its current status. So for a non-main task we mark it completed first,
        ensuring an injected task's status reflects the finished run.
        """
        self.detach_loguru()
        is_main = getattr(self.task, "is_main_task", lambda: True)()
        if not is_main and getattr(self.task, "status", None) not in _FINISHED_STATUSES:
            self.task.mark_completed()
        self.task.close()

    def __enter__(self) -> ClearMLTracker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
