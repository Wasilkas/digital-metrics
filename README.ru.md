# digital-metrics

Библиотека для оценки качества детекции объектов. Вычисляет метрики детекции
по классам (precision, recall, F1, mAP50 / mAP75 / mAP50-95, каппа Коэна,
доверительные интервалы Уилсона) на основе таблиц pandas DataFrame с эталонной
разметкой (ground truth) и предсказаниями модели. На выходе — объекты `Metrics`,
матрица ошибок, дашборды в Excel и графики доверительных интервалов.

> 🇬🇧 English version: [README.md](README.md)

---

## Установка

```bash
uv pip install git+https://github.com/Wasilkas/digital-metrics
# или
pip install git+https://github.com/Wasilkas/digital-metrics
```

Требуется Python 3.11+. Базовая установка не тянет `torch`; опциональные
бэкенды `ultralytics` / `torchmetrics` подключаются как extras — см.
[Внешние бэкенды метрик](#внешние-бэкенды-метрик-единая-точка-входа).

---

## Схема входных данных

Обе таблицы используют одинаковые имена столбцов:

| Столбец | Тип | GT | Preds | Описание |
|---|---|:---:|:---:|---|
| `image_name` | `str` | ✓ | ✓ | Уникальный идентификатор изображения |
| `instance_label` | `str` | ✓ | ✓ | Название класса |
| `bbox_x_tl` | `float` | ✓ | ✓ | Координата x верхнего-левого угла рамки |
| `bbox_y_tl` | `float` | ✓ | ✓ | Координата y верхнего-левого угла рамки |
| `bbox_x_br` | `float` | ✓ | ✓ | Координата x нижнего-правого угла рамки |
| `bbox_y_br` | `float` | ✓ | ✓ | Координата y нижнего-правого угла рамки |
| `split` | `str` | ✓ | — | `"train"` / `"val"` / `"test"` |
| `confidence` | `float` | — | ✓ | Уверенность детекции в диапазоне `[0, 1]` |
| `image_path` | `str` | опц. | — | Полный путь к файлу изображения; нужен **только** для `Evaluation.predict_to_dataframe` (инференс YOLO) |
| `image_width` | `int` | опц. | — | Ширина изображения в пикселях; нужна **только** при `skip_cohen_kappa=False` (пиксельные маски каппы Коэна) |
| `image_height` | `int` | опц. | — | Высота изображения в пикселях; нужна **только** при `skip_cohen_kappa=False` (пиксельные маски каппы Коэна) |

Где `GT` — таблица эталонной разметки, `Preds` — таблица предсказаний модели.

### Валидация входных данных

При запуске оценки входные данные проверяются, и поднимается `ValueError`, если:

- отсутствуют обязательные столбцы (по схеме выше);
- в столбце `confidence` таблицы предсказаний есть значения `NA`;
- метки `instance_label` предсказаний отсутствуют в наборе классов эталона.

(Схема калибровки val/test дополнительно отклоняет сплиты, которые делят общий
`image_name`, чтобы исключить утечку калибровочных данных.)

---

## Быстрый старт

```python
import pandas as pd
from metrics import Evaluation

preds_df = pd.read_csv("predictions.csv", index_col=0)
split_df = pd.read_csv("ground_truth.csv", index_col=0)

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", find_best_confs=True)

# Метрики по классам
for cls, m in ev.metrics.items():
    print(f"{cls}: P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1_score:.3f}  mAP50={m.ap50:.3f}")

# Пороги уверенности, подобранные для максимизации F1 по каждому классу
print(ev.best_confidences)

# Матрица ошибок
print(ev.cm)          # ndarray (n_classes+1, n_classes+1)
print(ev.class_labels)
```

---

## Калибровка порогов по валидации (рекомендуется)

Подберите оптимальные пороги уверенности на валидационном сплите, затем
оцените модель на тесте — это исключает «оптимизм» от подбора порогов на тех же
данных, на которых считаются метрики:

```python
ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", calibration_split="val")
```

Калибровочный и оцениваемый сплиты не должны пересекаться по `image_name` —
иначе возникнет утечка данных, и `Evaluation` поднимет `ValueError`.

---

## Оптимизация порога уверенности

Когда включён `find_best_confs=True` (или задан `calibration_split`), пороги
уверенности подбираются автоматически. Доступны два режима через параметр
`confidence_optimization`:

```python
from metrics import Evaluation, ConfidenceOptimization

# По умолчанию: отдельный порог для каждого класса, максимизирующий его F1
ev = Evaluation(preds_df, split_df, confidence_optimization="per_class")

# В стиле YOLO: единый порог, общий для всех классов
ev = Evaluation(preds_df, split_df, confidence_optimization="global")
ev(split="test", calibration_split="val")
```

- **`"per_class"`** (по умолчанию) — в `ev.best_confidences` для каждого класса
  свой порог, подобранный для максимума F1 этого класса. Подходит, когда нужно
  выжать максимум качества по каждому классу.
- **`"global"`** — повторяет поведение Ultralytics YOLO, который применяет
  **один** порог уверенности ко всем классам. Выбирается порог, максимизирующий
  **средний F1 по классам**, и применяется одинаково ко всем — поэтому все
  значения в `ev.best_confidences` совпадают. Используйте этот режим, когда
  метрики должны быть сопоставимы с YOLO, или когда в продакшене применяется
  единое значение `conf`.

Оба режима работают и в схеме калибровки по валидации: пороги находятся на
калибровочном сплите и применяются к оцениваемому.

> Если выбранный порог равен **минимальной** уверенности предсказаний, отсечение
> сохраняет все детекции — оптимизация ничего не дала (например, предсказания
> настолько совпадают с эталоном, что оптимальный по F1 порог — это «пол»). В этом
> случае пишется `WARNING` (по каждому классу, либо один раз для `"global"`).

---

## Стратегии сопоставления рамок

Сопоставление предсказанных и эталонных рамок (box matching) доступно в трёх
вариантах через параметр `matching_strategy`:

```python
from metrics import Evaluation, MatchingStrategy

# По умолчанию: iou_prior — в стиле Ultralytics (без scipy), сортировка по IoU
ev = Evaluation(preds_df, split_df, matching_strategy="iou_prior")

# greedy — сортировка по уверенности (в стиле YOLO)
ev = Evaluation(preds_df, split_df, matching_strategy="greedy")

# Hungarian — глобально оптимальное сопоставление по геометрии
ev = Evaluation(preds_df, split_df, matching_strategy="hungarian")
```

### `greedy`

1. Предсказания сортируются по убыванию уверенности.
2. Для каждого предсказания ищется ещё не занятая эталонная рамка с
   максимальным IoU.
3. Если IoU ≥ порога — эталонная рамка считается занятой; при совпадении классов
   это TP, иначе FP с реальной меткой эталона.
4. Иначе — FP с меткой эталона `"background"`.
5. Все несопоставленные эталонные рамки → FN.

Используется для P/R/F1/матрицы ошибок и во внутреннем цикле mAP.

### `iou_prior` (по умолчанию, в стиле Ultralytics, без scipy)

1. Находятся все пары «предсказание–эталон», где IoU ≥ порога **и** классы совпадают.
2. Пары сортируются по убыванию IoU.
3. Жадное назначение: каждое предсказание и каждый эталон используются не более
   одного раза; побеждает пара с наибольшим IoU.
4. Несопоставленные предсказания → FP.
5. Несопоставленные эталоны → FN.

Ключевое отличие от greedy: **уверенность не участвует в подборе пар**.
Предсказание с меньшей уверенностью, но лучшим IoU, забирает эталонную рамку.

### `hungarian` (глобально оптимальное)

Использует `scipy.optimize.linear_sum_assignment` на матрице отрицательных IoU.
Сначала геометрия, уверенность не учитывается. Дороже по времени — O(N³) на
изображение. Подходит для аудита разметки, когда нужно наиболее правдоподобное
сопоставление предсказанных и эталонных рамок.

Все три стратегии поддерживаются и в расчёте mAP.

---

## Методы интегрирования AP (`APMethod`)

Способ вычисления площади под P-R кривой задаётся параметром `ap_method`:

```python
from metrics import Evaluation, APMethod

# По умолчанию: 101-точечная интерполяция в стиле COCO / Ultralytics
ev = Evaluation(preds_df, split_df, ap_method="interp")

# Непрерывное интегрирование (VOC 2010+)
ev = Evaluation(preds_df, split_df, ap_method="continuous")
```

- **`"interp"`** (по умолчанию) — 101-точечная интерполяция COCO, совместимая с
  Ultralytics. Интегрирует через `np.trapezoid` по 101 равномерной точке recall.
- **`"continuous"`** — интегрирование по площади прямоугольников (VOC 2010+).
  Добавляет опорные точки `(0, 0)` и `(1, 0)`, строит огибающую precision справа
  налево, суммирует площади прямоугольников в точках изменения recall.

На фикстуре датасета методы расходятся не более чем на 0.001 по среднему mAP50.

---

## Предобработка предсказаний

Фильтрацию по уверенности и/или кастомный NMS можно включить, передав пороги
в конструктор:

```python
ev = Evaluation(
    preds_df,
    split_df,
    # Отбросить предсказания с низкой уверенностью
    preprocess_preds_conf_threshold=0.25,
    # Подавить рамку того же класса, в значительной степени вложенную в другую (containment >= 0.8)
    preprocess_preds_nms_containment_threshold=0.8,
    # Подавить рамку с меньшей уверенностью при пересечении рамок разных классов (IoU >= 0.5)
    preprocess_preds_nms_iou_threshold=0.5,
)
```

Каждый порог независим — задавайте только нужные. Значение `None` (по умолчанию)
отключает соответствующий тип подавления.

> Важно: предобработка применяется к `preds_df` (для P/R/F1/матрицы ошибок),
> но расчёт mAP всегда выполняется на исходных, необработанных предсказаниях
> (`_raw_preds_df`) — как и в архитектуре Ultralytics.

---

## Воспроизведение метрик YOLO (Ultralytics)

Чтобы получить значения, максимально близкие к запуску `model.val()` в
Ultralytics, повторите конфигурацию инференса модели и выберите YOLO-совместимые
опции:

```python
ev = Evaluation(
    preds_df,
    split_df,
    iou_threshold=0.5,                       # P/R/F1 при IoU 0.50 (рабочая точка mAP50)
    preprocess_preds_conf_threshold=0.001,   # та же минимальная уверенность, что в конфиге модели
    preprocess_preds_nms_iou_threshold=0.7,  # тот же порог NMS IoU, что в конфиге модели
    ap_method="interp",                      # 101-точечное интегрирование трапециями (COCO / Ultralytics)
    confidence_optimization="global",        # единый порог уверенности для всех классов
)
ev(split="test", calibration_split="val")
```

- **`preprocess_preds_conf_threshold` / `preprocess_preds_nms_iou_threshold`** —
  задайте значения `conf` и `iou` из конфигурации вашей YOLO (для валидации по
  умолчанию `conf=0.001`, `iou=0.7`), чтобы предсказания попадали в сопоставление
  в той же рабочей точке, что использовала модель. Если `preds_df` уже *выгружен*
  из модели (NMS уже применён), порог NMS можно оставить `None` — повторное
  применение служит лишь подстраховкой от межклассовых дубликатов.
- **`ap_method="interp"`** — 101-точечное интегрирование трапециями
  (`np.trapezoid`) — это в точности способ расчёта AP в Ultralytics. Значение по
  умолчанию в библиотеке.
- **`confidence_optimization="global"`** — YOLO применяет один порог уверенности
  ко всем классам, выбранный по кривой среднего F1.
- **`matching_strategy="iou_prior"`** — значение по умолчанию; повторяет
  внутреннее сопоставление Ultralytics по убыванию IoU. Альтернатива `"greedy"`
  (в стиле YOLO, сортировка по уверенности) расходится примерно на 0.006 mAP50
  на фикстуре.

### Почему другие параметры не создают проблемы

Библиотека намеренно гибче, чем YOLO, и отклонение от рецепта выше **не** делает
метрики неверными — оно лишь меняет угол зрения:

- **mAP не зависит от параметров рабочей точки.** `mAP50/75/50-95` всегда
  считаются на исходных, неотфильтрованных предсказаниях по *всей* кривой
  precision–recall, поэтому `preprocess_preds_conf_threshold`, пороги NMS и
  `confidence_optimization` **никак не влияют** на mAP. Эти параметры лишь сдвигают
  единственную точку, в которой отчитываются precision/recall/F1 и матрица ошибок —
  любой выбор даёт корректную рабочую точку на той же кривой.
- **Метод AP почти не важен.** `"interp"` (трапеции COCO, по умолчанию) и
  `"continuous"` (точная площадь прямоугольников VOC) расходятся на ≤ 0.001 на
  фикстуре; оба — стандартные определения, поэтому любой из них обоснован.
- **Per-class порог не хуже global.** Глобальный порог — это частный (ограниченный)
  случай per-class (одно общее значение против лучшего значения на класс), поэтому
  `"per_class"` даёт не меньший средний F1 и более тонкую рабочую точку — не
  затрагивая mAP.
- **Все стратегии сопоставления — допустимые правила назначения.** greedy /
  iou_prior / hungarian различаются по mAP лишь незначительно; выбирайте по задаче
  (greedy/iou_prior — для совпадения с YOLO, hungarian — для аудита разметки).

Коротко: используйте рецепт, когда нужны цифры, напрямую сравнимые с запуском
Ultralytics; в остальных случаях значения по умолчанию (или per-class порог) дают
более богатую и зачастую более выгодную картину, сохраняя mAP столь же сравнимым.

---

## Внешние бэкенды метрик (единая точка входа)

Чтобы получить значения из устоявшейся библиотеки метрик — вместо собственного
пути `Evaluation` — используйте единую точку входа `compute_detection_metrics`.
Она считает метрики по тем же таблицам GT/предсказаний через один из двух
опциональных бэкендов и возвращает `dict[str, DetectionMetrics]` (по классам:
`precision / recall / f1 / ap50 / ap75 / ap50_95`):

```python
from metrics import compute_detection_metrics

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
в [docs/why_prf1_differs.md](docs/why_prf1_differs.md) (на английском).

---

## `Evaluation` с внешним бэкендом

Те же два бэкенда встроены в `Evaluation`: можно выбрать движок метрик и
сохранить остальной рабочий процесс — дашборды, графики CI, матрицу ошибок.
Передайте `backend=` в конструктор или вызовите бэкенд напрямую:

```python
from metrics import Evaluation

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
  сплите (in-sample). Передайте `calibration_split="val"`, и бэкенд
  `"ultralytics"` будет отчитывать P/R/F1 в точке F1-оптимальной уверенности,
  найденной на `val`, считывая её по per-class кривым `ap_per_class`; **AP
  остаётся по всей кривой** (по-прежнему совпадает с `model.val()`), а выбранные
  пороги попадают в `ev.best_confidences`. `confidence_optimization` выбирает
  `"per_class"` или `"global"` пороги — как и в нативном пути. `"torchmetrics"`
  игнорирует `calibration_split` с предупреждением (пока не поддерживается).
  Те же механизмы доступны отдельно: `find_ultralytics_confidence(gt_df, preds_df,
  mode=...)` и `compute_ultralytics_metrics(..., conf_threshold=...)`.

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

---

## От весов YOLO к предсказаниям

Если у вас есть модель Ultralytics, а не готовая таблица предсказаний, запустите
инференс прямо из таблицы эталона — `Evaluation.predict_to_dataframe` замыкает
конвейер оценки с начала, без `data.yaml`:

```python
from metrics import Evaluation

# В эталоне должен быть столбец `image_path` (полный путь к каждому изображению).
# Создайте объект с preds_df=None, затем сгенерируйте предсказания моделью:
ev = Evaluation(None, "ground_truth.csv", iou_threshold=0.5)
ev.predict_to_dataframe("best.pt", split="val")   # заполняет ev.preds_df
ev(split="val")                                    # обычная оценка
```

- Источник изображений — `split_df["image_path"]`; `image_name` — это последняя
  часть пути (`Path(image_path).name`), поэтому предсказания автоматически
  стыкуются с эталоном. `instance_label` берётся из имён классов модели; рамки —
  в пикселях `xyxy`.
- `split=` задаёт, по каким изображениям запускать инференс: один сплит (`"val"`),
  список сплитов (`["test", "val"]`) или `None` — по всем изображениям в
  `split_df`. (При авто-генерации предсказаний из `weights_path` `Evaluation`
  делает это сам — запуская только оцениваемый сплит плюс `calibration_split`,
  если он задан.)
- Модель запускается с `conf=0.001`, `iou=0.7` по умолчанию (как в YOLO val), чтобы
  ниже по конвейеру была доступна вся кривая precision-recall; поднимите `conf=`
  для предварительной фильтрации.
- Любые дополнительные аргументы `model.predict` передаются напрямую как
  именованные аргументы —
  `ev.predict_to_dataframe("best.pt", split="val", imgsz=1280, half=True, augment=True)`
  — либо в режиме авто-предсказания через конструктор:
  `predict_kwargs={"imgsz": 1280, "half": True}`.
- `predict_to_dataframe` также **возвращает** DataFrame с предсказаниями — его можно
  сохранить (`df.to_csv(...)`) или передать в `compute_detection_metrics`.
- `image_name=` задаёт формат `image_name` (`"name"` — имя файла с расширением, по
  умолчанию; `"stem"`; либо полный `"path"`) — согласуйте с `image_name` эталона.

Требуется extra `ultralytics` (ленивый импорт; базовая установка остаётся без
`torch`).

---

## Результаты

### `ev.metrics` — `dict[str, Metrics]`

Каждый объект `Metrics` содержит:

| Поле | Описание |
|---|---|
| `tp`, `fp`, `fn` | Истинно-положительные / ложно-положительные / ложно-отрицательные |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | 2 · P · R / (P + R) |
| `perebrak` | 1 − precision (доля ложных срабатываний; доменный термин) |
| `nedobrak` | 1 − recall (доля пропусков; доменный термин) |
| `ap50` | AP при IoU = 0.50 (`nan`, если класс отсутствует в сплите) |
| `ap75` | AP при IoU = 0.75 (`nan`, если класс отсутствует в сплите) |
| `ap50_95` | mAP, усреднённый по IoU 0.50 … 0.95 (`nan`, если класс отсутствует) |
| `cohen_kappa` | Каппа Коэна (метод пиксельных масок) |
| `confidence` | Лучший порог уверенности для данного класса |
| `precision_ci_lower/upper` | 95 % доверительный интервал Уилсона для precision |
| `recall_ci_lower/upper` | 95 % доверительный интервал Уилсона для recall |
| `perebrak_ci_lower/upper` | Доверительный интервал для perebrak |
| `nedobrak_ci_lower/upper` | Доверительный интервал для nedobrak |

### Дашборды и графики

```python
# Дашборды в Excel + опциональное изображение матрицы ошибок
summary_df, detail_df = ev.get_dashboards(
    save_to_excel=True,
    path="/path/to/output/",
    save_confusion_matrix=True,
)

# Столбчатая диаграмма доверительных интервалов
fig, ax = ev.plot_confidence_intervals(
    metric="precision",         # или "recall", "perebrak", "nedobrak"
    confidence_level=0.95,
    save_path="/path/to/ci_plot.png",
)
```

Каталоги для вывода (`path` / родительский каталог `save_path`) создаются
автоматически, если их ещё нет.

### Аудит ошибок

```python
# Top-k пар предсказание/эталон, перепутанных между двумя классами
audit_df = ev.get_topk_confusions(main_class="car", k=20)

# DataFrame с разметкой типа сопоставления для визуализации
gt_vis, pred_vis = ev.get_dfs_visualization()
```

---

## Конструктор `Evaluation`

```python
Evaluation(
    preds_df: pd.DataFrame | str | None,   # DataFrame, путь к CSV или None — чтобы сначала предсказать
    split_df: pd.DataFrame | str,
    iou_threshold: float = 0.5,
    preprocess: bool = False,        # удалять почти идентичные дубликаты эталонных рамок
    skip_cohen_kappa: bool = True,   # каппа дорогая; включайте только при необходимости
    matching_strategy: MatchingStrategy = "iou_prior",  # "iou_prior" | "greedy" | "hungarian"
    preprocess_preds_conf_threshold: float | None = None,
    preprocess_preds_nms_containment_threshold: float | None = None,
    preprocess_preds_nms_iou_threshold: float | None = None,
    ap_method: APMethod = "interp",                              # "interp" | "continuous"
    confidence_optimization: ConfidenceOptimization = "per_class",  # "per_class" | "global"
    weights_path: str | None = None,   # веса YOLO для авто-предсказания, когда preds_df=None
    backend: Backend | None = None,    # None = нативный путь; "ultralytics" | "torchmetrics"
    predict_kwargs: dict | None = None,  # доп. аргументы model.predict(...) для прогона от весов
)
```

Значения по умолчанию выбраны в стиле YOLO (`matching_strategy="iou_prior"`,
`ap_method="interp"`).

`preds_df` / `split_df` принимают как DataFrame, так и путь к CSV-файлу. Передайте
`preds_df=None` вместе с `weights_path`, чтобы прогнать **весь конвейер от весов**:
первый вызов сгенерирует предсказания моделью только по тем сплитам, которые
будут использованы (оцениваемый сплит плюс `calibration_split`, если задан), и
затем выполнит оценку:

```python
ev = Evaluation(None, "ground_truth.csv", weights_path="best.pt")
ev(split="val")   # предсказывает из best.pt, затем оценивает
```

Если `preds_df=None`, а `weights_path` не задан, вызов оценки поднимает
`ValueError`. Можно также сначала предсказать вручную через
[`predict_to_dataframe`](#от-весов-yolo-к-предсказаниям).

Набор изображений для каждого сплита определяется автоматически по столбцу
`split` в `split_df` — отдельный список передавать не нужно.

- `iou_threshold` — порог IoU для сопоставления рамок (P/R/F1/CM).
- `preprocess` — удалить дублирующиеся эталонные рамки (по почти идентичному IoU).
- `skip_cohen_kappa` — пропустить расчёт каппы Коэна (он требует столбцов
  `image_width` / `image_height` и заметно медленнее).
- `preprocess_preds_conf_threshold` — отбросить предсказания с уверенностью
  строго ниже порога. `None` отключает фильтрацию.
- `preprocess_preds_nms_containment_threshold` — подавление вложенности внутри
  класса: из пары рамок одного класса удаляется рамка с меньшей уверенностью,
  когда `intersection / min(area_a, area_b) >= threshold` (одна рамка почти
  целиком внутри другой). `None` отключает.
- `preprocess_preds_nms_iou_threshold` — межклассовое подавление по IoU: при
  `IoU >= threshold` для рамок разных классов удаляется рамка с меньшей
  уверенностью. `None` отключает.
- `ap_method` — метод интегрирования AP: `"interp"` (по умолчанию) или
  `"continuous"`.
- `confidence_optimization` — `"per_class"` (по умолчанию) подбирает порог для
  каждого класса; `"global"` выбирает единый порог в стиле YOLO, общий для всех
  классов (см. раздел [Оптимизация порога уверенности](#оптимизация-порога-уверенности)).
- `backend` — `None` (по умолчанию) запускает нативный конвейер; `"ultralytics"` /
  `"torchmetrics"` заставляют `Evaluation` считать метрики сплита через
  соответствующую внешнюю библиотеку (см. раздел
  [`Evaluation` с внешним бэкендом](#evaluation-с-внешним-бэкендом)).
- `predict_kwargs` — дополнительные аргументы, передаваемые в `model.predict`
  Ultralytics при авто-генерации предсказаний из `weights_path` (например,
  `{"conf": 0.25, "imgsz": 1280, "half": True, "augment": True}`). Игнорируется,
  если задан `preds_df`. Для разового запуска те же аргументы можно передать прямо
  в [`predict_to_dataframe`](#от-весов-yolo-к-предсказаниям).

### Группирующие конфиги (опционально)

Чтобы не передавать десяток плоских аргументов, конструктор также принимает три
опциональных группирующих конфига. Они **полностью аддитивны** — все плоские
аргументы выше продолжают работать, — и каждая группа, если передана, задаёт
целиком свою группу и имеет приоритет над соответствующими плоскими аргументами:

```python
from metrics import Evaluation, ScoringConfig, PreprocessConfig, InferenceConfig

ev = Evaluation(
    preds_df,
    split_df,
    scoring=ScoringConfig(iou_threshold=0.5, matching_strategy="greedy"),
    preprocessing=PreprocessConfig(conf_threshold=0.25, nms_iou_threshold=0.5),
    inference=InferenceConfig(weights_path="best.pt", predict_kwargs={"imgsz": 1280}),
)
```

- **`ScoringConfig`** — `iou_threshold`, `matching_strategy`, `ap_method`,
  `confidence_optimization`, `skip_cohen_kappa`.
- **`PreprocessConfig`** — `dedup_gt` (плоский `preprocess`), `conf_threshold`,
  `nms_containment_threshold`, `nms_iou_threshold`.
- **`InferenceConfig`** — `weights_path`, `predict_kwargs`.

Значения по умолчанию у конфигов совпадают с плоскими, поэтому
`Evaluation(preds, split)` и `Evaluation(preds, split, scoring=ScoringConfig())`
ведут себя одинаково. `backend` остаётся плоским аргументом верхнего уровня.

### Вызов `evaluation(...)`

```python
ev(
    split="all",                  # "all" / "train" / "val" / "test"
    find_best_confs=True,         # подбирать пороги по F1 (in-sample, если нет calibration_split)
    calibration_split=None,       # например "val" — подобрать пороги на этом сплите
)
```

### Доступные атрибуты после вызова

- `ev.metrics` — `dict[str, Metrics]`
- `ev.cm`, `ev.class_labels` — матрица ошибок и подписи классов (`ev.cm` равно
  `None` в режиме бэкенда `"torchmetrics"`)
- `ev.detection_metrics` — `dict[str, DetectionMetrics]`, «сырой» результат
  внешнего бэкенда; заполняется только в режиме `backend` (иначе пустой)
- `ev.best_confidences` — `dict[str, float]`, оптимальный порог по каждому классу
- `ev.unfiltered_matches` — сопоставления до отсечения по уверенности

---

## Определения метрик

### Стандартные метрики (совместимы с YOLO)

- **IoU** — `площадь пересечения / площадь объединения`
- **TP** — IoU ≥ порога, корректная метка, эталон ещё не занят (одно сопоставление на эталон)
- **FP** — нет сопоставленного эталона (IoU ниже порога или все кандидаты заняты)
- **FN** — эталонная рамка без сопоставленного предсказания
- **Precision** — `TP / (TP + FP)`
- **Recall** — `TP / (TP + FN)`
- **F1** — `2 · P · R / (P + R)`
- **perebrak** — `1 − precision` (доменный термин; доля ложных срабатываний)
- **nedobrak** — `1 − recall` (доменный термин; доля пропусков)
- **CI** — доверительный интервал Уилсона для precision / recall

### mAP

mAP вычисляется **независимо** от посчёта P/R/F1 на одном пороге. Используется
исходная (необработанная) таблица предсказаний, и для каждого порога IoU
запускается отдельный внутренний цикл сопоставления — как в двухпутевой
архитектуре Ultralytics:

- Все предсказания сортируются по убыванию уверенности (глобально по классу).
- Для каждого порога IoU из `[0.50, 0.55, …, 0.95]` (10 значений):
  - Сопоставление выполняется выбранной стратегией (greedy, iou_prior или hungarian).
  - Накапливаются кумулятивные TP/FP → строится P-R кривая.
- **AP** — площадь под P-R кривой (метод задаётся `ap_method`).
- **mAP50** — AP при IoU = 0.50
- **mAP75** — AP при IoU = 0.75
- **mAP50-95** — среднее AP по всем 10 порогам

Классы **без эталонных экземпляров в оцениваемом сплите** получают `nan` для
`ap50`, `ap75` и `ap50_95` (а не `0.0`), чтобы `nanmean` корректно исключал
отсутствующие классы из усреднения.

### Что НЕ совпадает с YOLO один в один

Precision, recall, F1 и матрица ошибок считаются через сопоставление на одном
пороге IoU с опциональной фильтрацией по уверенности и классификацией TP с
учётом меток. YOLO не выдаёт P/R/F1 по порогам таким же образом — эти метрики
намеренно кастомные и не должны численно совпадать с выводом YOLO в консоли.

---

## Разработка

```bash
git clone https://github.com/Wasilkas/digital-metrics
cd digital-metrics
uv venv && uv sync

uv run ruff check . --fix     # линтинг
uv run ruff format .          # форматирование
uv run mypy src/              # проверка типов
uv run pytest --cov=src/metrics tests/   # тесты с покрытием
```
