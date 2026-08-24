[← Оглавление документации](README.ru.md) · [🇬🇧 English version](inference.md)

# Инференс YOLO

Если у вас есть модель Ultralytics, а не готовая таблица предсказаний, запустите
инференс прямо из таблицы эталона — `Evaluation.predict_to_dataframe` замыкает
конвейер оценки с начала, без `data.yaml`:

```python
from digital_metrics import Evaluation

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
- **Память GPU** — инференс идёт чанками по `batch` изображений (по умолчанию 16),
  поэтому пик VRAM ограничен (≈ `batch` × стоимость одной картинки) и не растёт с
  числом изображений. Если прогон падает по памяти, сначала уменьшите `batch`,
  затем `imgsz` и/или включите `half=True` — например,
  `predict_kwargs={"batch": 4, "imgsz": 1280, "half": True}`. (`batch` здесь —
  настоящий размер чанка; штатный аргумент `batch` в `model.predict` в
  стриминг-режиме не работает.)
- `predict_to_dataframe` также **возвращает** DataFrame с предсказаниями — его можно
  сохранить (`df.to_csv(...)`) или передать в `compute_detection_metrics`.
- `image_name=` задаёт формат `image_name` (`"name"` — имя файла с расширением, по
  умолчанию; `"stem"`; либо полный `"path"`) — согласуйте с `image_name` эталона.

Требуется extra `ultralytics` (ленивый импорт; базовая установка остаётся без
`torch`).

