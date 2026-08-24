[← Оглавление документации](README.ru.md) · [🇬🇧 English version](backends.md)

# Внешние бэкенды метрик

Чтобы получить значения из устоявшейся библиотеки метрик — вместо собственного
пути `Evaluation` — используйте единую точку входа `compute_detection_metrics`.
Она считает метрики по тем же таблицам GT/предсказаний через один из двух
опциональных бэкендов и возвращает `dict[str, DetectionMetrics]` (по классам:
`precision / recall / f1 / ap50 / ap75 / ap50_95`):

```python
from digital_metrics import compute_detection_metrics

gt_df = split_df[split_df["split"] == "test"]

# Сравнимо с YOLO (собственный ap_per_class из Ultralytics)
yolo = compute_detection_metrics(gt_df, preds_df, backend="ultralytics")

# Общий COCO mAP (MeanAveragePrecision из torchmetrics)
coco = compute_detection_metrics(gt_df, preds_df, backend="torchmetrics")

for cls, m in yolo.items():
    print(f"{cls}: P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f} "
          f"mAP50={m.ap50:.3f} mAP50-95={m.ap50_95:.3f}")
```

- **`backend="ultralytics"`** (по умолчанию) — сравнимо с YOLO. Рамки
  сопоставляются и оцениваются собственным `ap_per_class` из Ultralytics, поэтому
  AP совпадает с `model.val()`. P/R/F1 считываются при IoU 0.50 в единой
  глобальной рабочей точке максимума среднего F1.
- **`backend="torchmetrics"`** — общий COCO mAP через `MeanAveragePrecision`
  из torchmetrics (pycocotools). AP — это собственные `map_50 / map_75 / map`
  torchmetrics по классам; P/R/F1 выводятся по его P-R кривой при IoU 0.50 в
  точке максимума F1 для каждого класса (своих готовых P/R/F1 у torchmetrics нет).

Оба бэкенда оценивают только классы, у которых есть хотя бы одна эталонная рамка
в сплите. Каждый — тяжёлый **опциональный extra** (оба тянут `torch`) с ленивым
импортом, поэтому базовая установка остаётся без `torch`. Установите нужный:

```bash
# из клонированного репозитория
uv sync --extra ultralytics
uv sync --extra torchmetrics

# или напрямую
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
```

Вызов бэкенда без установленного extra поднимает `ImportError` с подсказкой по
установке; неизвестный `backend` — `ValueError`. Базовые функции
(`compute_ultralytics_metrics`, `compute_torchmetrics_metrics`) тоже публичны и
вызываются напрямую. `YoloMetrics` сохранён как обратносовместимый алиас
`DetectionMetrics`.

> Эти бэкенды — путь для прямого сравнения «один в один». Собственные P/R/F1 из
> `Evaluation` намеренно кастомные и **не** предназначены для численного
> совпадения с выводом YOLO (см. примечание выше).

На фикстуре все три способа сходятся по mAP до ~0.002–0.006, но расходятся по
P/R/F1 до ~0.05 — это структурное следствие того, как с одной и той же кривой
выбирается и считывается одна рабочая точка (порог на класс против единого
глобального; «сырая» precision против огибающей COCO). Объяснение с графиками —
в [docs/why_prf1_differs.md](why_prf1_differs.md) (на английском).

---

## `Evaluation` с внешним бэкендом

Те же два бэкенда встроены в `Evaluation`: можно выбрать движок метрик и
сохранить остальной рабочий процесс — дашборды, графики CI, матрицу ошибок.
Передайте `backend=` в конструктор или вызовите бэкенд напрямую:

```python
from digital_metrics import Evaluation

ev = Evaluation(preds_df, split_df, backend="ultralytics")  # или "torchmetrics"
ev(split="test")

ev.detection_metrics   # «сырой» dict[str, DetectionMetrics] от бэкенда
ev.metrics             # те же числа, адаптированные к нативным Metrics
ev.get_dashboards()    # работает — построено по результатам бэкенда

# Либо запустить бэкенд, не переключая весь Evaluation:
yolo = ev.compute_metrics_ultralytics(split="test")
coco = ev.compute_metrics_torchmetrics(split="test")
```

- `backend=None` (по умолчанию) запускает нативный конвейер. `"ultralytics"` /
  `"torchmetrics"` считают метрики сплита по **исходным** предсказаниям (как
  `model.val()`); `find_best_confs` и пороги предобработки в этом режиме не
  применяются.
- **Калибровка** — по умолчанию бэкенд сам выбирает рабочую точку на оцениваемом
  сплите (in-sample). Передайте `calibration_split="val"`, и бэкенд будет
  отчитывать P/R/F1 в точке F1-оптимальной уверенности, найденной на `val`,
  считывая её по своим per-class кривым; **AP остаётся по всей кривой**, а
  выбранные пороги попадают в `ev.best_confidences`. `confidence_optimization`
  выбирает `"per_class"` или `"global"` пороги — как и в нативном пути.
  **Поддерживают оба бэкенда** — `"ultralytics"` читает по кривым `ap_per_class`,
  `"torchmetrics"` — по кривым precision/score при IoU 0.50 из `extended_summary`.
  Те же механизмы доступны отдельно: `find_ultralytics_confidence` /
  `find_torchmetrics_confidence` (с `mode=...`) и
  `compute_*_metrics(..., conf_threshold=...)`.

  ```python
  ev = Evaluation(preds_df, split_df, backend="ultralytics",
                  confidence_optimization="per_class")
  ev(split="test", calibration_split="val")   # калибровка на val, отчёт на test
  ```
- `ev.detection_metrics` хранит нетронутый вывод бэкенда; `ev.metrics` — те же
  precision / recall / f1 / AP, **адаптированные к нативным `Metrics`**: TP/FP/FN
  восстанавливаются как дробные числа из количества эталонных рамок класса, чтобы
  дашборды и графики CI продолжали работать. В этом режиме `cohen_kappa` равен
  `-1`, а порог `confidence` по классу — `0.0`, если его не задал
  `calibration_split`.
- **Матрица ошибок** — бэкенд `"ultralytics"` заполняет `ev.cm` / `ev.class_labels`
  собственной логикой Ultralytics (numpy-порт `ConfusionMatrix.process_batch` с
  дефолтами conf 0.25 / IoU 0.45 — матрица, которую рисует `model.val()`),
  транспонированной к принятой здесь ориентации строка = эталон / столбец =
  предсказание. У `"torchmetrics"` матрицы ошибок нет, поэтому `ev.cm` равно
  `None`, и `get_dashboards` пропускает этот лист. Отдельная функция
  `compute_ultralytics_confusion_matrix(gt_df, preds_df)` также публична.

