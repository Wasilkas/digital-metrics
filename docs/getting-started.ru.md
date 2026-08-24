[← Оглавление документации](README.ru.md) · [🇬🇧 English version](getting-started.md)

# Начало работы

```bash
uv pip install git+https://github.com/Wasilkas/digital-metrics
# или
pip install git+https://github.com/Wasilkas/digital-metrics
```

Требуется Python 3.11+. Базовая установка не тянет `torch`. Опциональные extras
добавляют бэкенды метрик `ultralytics` / `torchmetrics` (см.
[Внешние бэкенды метрик](backends.ru.md)) и слой
трекинга экспериментов `clearml` (см.
[Трекинг экспериментов (ClearML)](tracking.ru.md)):

```bash
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[clearml]      @ git+https://github.com/Wasilkas/digital-metrics"
```

`clearml` не тянет `torch`; оба бэкенда метрик тянут `torch`.

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
- в столбце `confidence` таблицы предсказаний есть значения `NA`.

Метки `instance_label` предсказаний, отсутствующие в наборе классов эталона, —
не ошибка: о них выводится предупреждение, и такие строки отбрасываются до
расчёта (метрики определены только для классов эталона). Если они отсутствуют
лишь потому, что модель выдаёт транслитерированные названия, см.
[Транслитерированные метки модели](transliteration.ru.md).

(Схема калибровки val/test дополнительно отклоняет сплиты, которые делят общий
`image_name`, чтобы исключить утечку калибровочных данных.)

---

## Быстрый старт

```python
import pandas as pd
from digital_metrics import Evaluation

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

