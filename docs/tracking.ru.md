[← Оглавление документации](README.ru.md) · [🇬🇧 English version](tracking.md)

# Трекинг экспериментов (ClearML)

`ClearMLTracker` переносит завершённый `Evaluation` в задачу
[ClearML](https://clear.ml) — скаляры, артефакты, графики и логи, — чтобы прогоны
версионировались и сравнивались в UI ClearML. Это **самостоятельный слой** поверх
`Evaluation`: сам код оценки ничего не знает про ClearML, а `clearml` — опциональный
extra без `torch`, импортируемый лениво.

```bash
uv pip install "digital-metrics[clearml] @ git+https://github.com/Wasilkas/digital-metrics"
```

```python
from digital_metrics import Evaluation, ClearMLTracker

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)

with ClearMLTracker(project_name="detector", task_name="run-42") as tracker:
    ev(split="test", calibration_split="val")
    tracker.log_evaluation(ev)     # скаляры + артефакты + графики + логи
```

`log_evaluation(ev, *, iteration=0, artifacts_dir=None, save_to_excel=True,
save_confusion_matrix=True)` один раз запускает `get_dashboards` и переносит четыре
вещи:

- **Скаляры** — P/R/F1/mAP по классам как скалярные графики, таблицу метрик по
  классам и средние по датасету (`mean_*`, с nan-безопасным усреднением для AP) как
  одиночные значения.
- **Артефакты** — DataFrame'ы дашбордов (аналитический/продакшен), `best_confidences`,
  матрицу ошибок и Excel-файлы, которые записал `get_dashboards`.
- **Графики** — четыре PNG доверительных интервалов как изображения и матрицу ошибок
  как CM-график.
- **Логи** — логи прогона через `loguru`-сток, подключённый к консоли ClearML.

```python
ClearMLTracker(
    task=None,                 # передать существующий clearml.Task или дать создать Task.init
    *,
    project_name="detector",   # используется только при создании задачи
    task_name="run-42",
    output_uri=None,           # куда ClearML складывает артефакты/модели
    attach_logs=True,          # подключить сток loguru → консоль ClearML
    log_level="INFO",
    **task_init_kwargs,        # передаётся в Task.init
)
```

- `clearml` импортируется **только** когда трекер сам создаёт задачу, поэтому при
  передаче готового `task=` extra не нужен (удобно в тестах).
- Отдельные слои тоже публичны: `log_scalars` / `log_artifacts` / `log_plots` /
  `attach_loguru` / `detach_loguru` / `close`. Как контекстный менеджер закрывает
  задачу на выходе.
- `summarize_metrics(metrics) -> (df_по_классам, means)` — используемый им
  `torch`/ClearML-независимый помощник (таблица по классам + nan-безопасные средние);
  он публичен и вызывается на любом `dict[str, Metrics]` отдельно.

