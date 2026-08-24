[← Documentation index](README.md) · [🇷🇺 Русская версия](tracking.ru.md)

# Experiment tracking (ClearML)

`ClearMLTracker` mirrors a finished `Evaluation` into a
[ClearML](https://clear.ml) task — scalars, artifacts, plots and logs — so runs
are versioned and comparable in the ClearML UI. It is a **standalone layer** that
sits on top of `Evaluation`; the core evaluation code knows nothing about ClearML,
and `clearml` is an optional, `torch`-free extra imported lazily.

```bash
uv pip install "digital-metrics[clearml] @ git+https://github.com/Wasilkas/digital-metrics"
```

```python
from digital_metrics import Evaluation, ClearMLTracker

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)

with ClearMLTracker(project_name="detector", task_name="run-42") as tracker:
    ev(split="test", calibration_split="val")
    tracker.log_evaluation(ev)     # scalars + artifacts + plots + logs
```

`log_evaluation(ev, *, iteration=0, artifacts_dir=None, save_to_excel=True,
save_confusion_matrix=True)` runs `get_dashboards` once and mirrors four things:

- **Scalars** — per-class P/R/F1/mAP as scalar plots, the per-class metrics table,
  and headline means (`mean_*`, nan-aware for AP) as single values.
- **Artifacts** — the analyst/production dashboard DataFrames, `best_confidences`,
  the confusion matrix, and any Excel files `get_dashboards` wrote.
- **Plots** — the four confidence-interval PNGs as images and the confusion matrix
  as a CM plot.
- **Logs** — run logs, via a `loguru` sink attached to the ClearML console.

```python
ClearMLTracker(
    task=None,                 # inject an existing clearml.Task, or let it Task.init one
    *,
    project_name="detector",   # used only when it creates the task
    task_name="run-42",
    output_uri=None,           # where ClearML stores artifacts/models
    attach_logs=True,          # install the loguru → ClearML console sink
    log_level="INFO",
    **task_init_kwargs,        # forwarded to Task.init
)
```

- `clearml` is imported **only** when the tracker creates its own task, so passing
  an existing `task=` needs no extra installed (handy in tests).
- The individual layers are also public: `log_scalars` / `log_artifacts` /
  `log_plots` / `attach_loguru` / `detach_loguru` / `close`. Used as a context
  manager, it closes the task on exit.
- `summarize_metrics(metrics) -> (per_class_df, means)` is the `torch`/ClearML-free
  helper it uses to build the per-class table and nan-aware means; it is public and
  callable on any `dict[str, Metrics]` on its own.

