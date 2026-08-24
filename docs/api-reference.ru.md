[← Оглавление документации](README.ru.md) · [🇬🇧 English version](api-reference.md)

# Справочник по API

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
    transliterated_labels: bool = False, # модель предсказывает транслитерированные названия классов
    translit_match_cutoff: float = 0.8,  # порог нечёткого сопоставления при восстановлении меток
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
[`predict_to_dataframe`](inference.ru.md).

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
  классов (см. раздел [Оптимизация порога уверенности](calibration.ru.md#оптимизация-порога-уверенности)).
- `backend` — `None` (по умолчанию) запускает нативный конвейер; `"ultralytics"` /
  `"torchmetrics"` заставляют `Evaluation` считать метрики сплита через
  соответствующую внешнюю библиотеку (см. раздел
  [`Evaluation` с внешним бэкендом](backends.ru.md#evaluation-с-внешним-бэкендом)).
- `predict_kwargs` — дополнительные аргументы, передаваемые в `model.predict`
  Ultralytics при авто-генерации предсказаний из `weights_path` (например,
  `{"conf": 0.25, "imgsz": 1280, "half": True, "augment": True}`). Игнорируется,
  если задан `preds_df`. Для разового запуска те же аргументы можно передать прямо
  в [`predict_to_dataframe`](inference.ru.md).
- `transliterated_labels` / `translit_match_cutoff` — восстановление
  транслитерированных названий классов по словарю эталона (см. раздел
  [Транслитерированные метки модели](transliteration.ru.md)).

## Группирующие конфиги (опционально)

Чтобы не передавать десяток плоских аргументов, конструктор также принимает три
опциональных группирующих конфига. Они **полностью аддитивны** — все плоские
аргументы выше продолжают работать, — и каждая группа, если передана, задаёт
целиком свою группу и имеет приоритет над соответствующими плоскими аргументами:

```python
from digital_metrics import Evaluation, ScoringConfig, PreprocessConfig, InferenceConfig

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

## Вызов `evaluation(...)`

```python
ev(
    split="all",                  # "all" / "train" / "val" / "test"
    find_best_confs=True,         # подбирать пороги по F1 (in-sample, если нет calibration_split)
    calibration_split=None,       # например "val" — подобрать пороги на этом сплите
)
```

## Доступные атрибуты после вызова

- `ev.metrics` — `dict[str, Metrics]`
- `ev.cm`, `ev.class_labels` — матрица ошибок и подписи классов (`ev.cm` равно
  `None` в режиме бэкенда `"torchmetrics"`)
- `ev.detection_metrics` — `dict[str, DetectionMetrics]`, «сырой» результат
  внешнего бэкенда; заполняется только в режиме `backend` (иначе пустой)
- `ev.best_confidences` — `dict[str, float]`, оптимальный порог по каждому классу
- `ev.unfiltered_matches` — сопоставления до отсечения по уверенности

