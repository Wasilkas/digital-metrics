# digital-metrics

Библиотека оценки детекции объектов. Считает метрики по классам (precision,
recall, F1, mAP50 / mAP75 / mAP50-95, каппа Коэна, доверительные интервалы
Уилсона) из pandas DataFrame с эталонными и предсказанными рамками. На выходе —
объекты `Metrics`, матрица ошибок, дашборды в Excel и графики CI.

> 🇬🇧 English version: [README.md](README.md)

---

## Установка

```bash
uv pip install git+https://github.com/Wasilkas/digital-metrics
# или
pip install git+https://github.com/Wasilkas/digital-metrics
```

Требуется Python 3.11+. Базовая установка не тянет `torch`. Опциональные extras
добавляют бэкенды метрик `ultralytics` / `torchmetrics` и слой трекинга
экспериментов `clearml`:

```bash
uv pip install "digital-metrics[ultralytics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[torchmetrics] @ git+https://github.com/Wasilkas/digital-metrics"
uv pip install "digital-metrics[clearml]      @ git+https://github.com/Wasilkas/digital-metrics"
```

`clearml` не требует `torch`; оба бэкенда метрик тянут `torch`.

---

## Быстрый старт

Оба DataFrame используют одинаковые столбцы: `image_name`, `instance_label`,
`bbox_x_tl`, `bbox_y_tl`, `bbox_x_br`, `bbox_y_br` (+ `split` в эталоне,
`confidence` в предсказаниях). Полная схема — в
[docs/getting-started.ru.md](docs/getting-started.ru.md).

```python
import pandas as pd
from digital_metrics import Evaluation

preds_df = pd.read_csv("predictions.csv", index_col=0)
split_df = pd.read_csv("ground_truth.csv", index_col=0)

ev = Evaluation(preds_df, split_df, iou_threshold=0.5)
ev(split="test", calibration_split="val")   # калибровка на val, отчёт по test

for cls, m in ev.metrics.items():
    print(f"{cls}: P={m.precision:.3f}  R={m.recall:.3f}  F1={m.f1_score:.3f}  mAP50={m.ap50:.3f}")

print(ev.best_confidences)     # пороги уверенности по классам
print(ev.cm, ev.class_labels)  # матрица ошибок
ev.get_dashboards(save_to_excel=True, path="dashboards")
```

---

## Документация

Полная документация — в [docs/](docs/README.ru.md).

**Руководства**

- [Начало работы](docs/getting-started.ru.md) — установка, схема данных, валидация, быстрый старт
- [Калибровка порогов уверенности](docs/calibration.ru.md) — калибровка на валидации, по классам и глобально
- [Стратегии сопоставления рамок](docs/matching.ru.md) — `iou_prior` / `greedy` / `hungarian`
- [Предобработка предсказаний](docs/preprocessing.ru.md) — фильтр по уверенности + кастомный NMS
- [Транслитерированные метки модели](docs/transliteration.ru.md) — восстановление меток эталона по транслиту
- [Воспроизведение метрик YOLO](docs/yolo-parity.ru.md) — настройки, совпадающие с Ultralytics
- [Внешние бэкенды метрик](docs/backends.ru.md) — `ultralytics` / `torchmetrics`
- [Инференс YOLO](docs/inference.ru.md) — предсказания прямо из весов
- [Результаты](docs/outputs.ru.md) — `Metrics`, дашборды, графики CI, аудит ошибок
- [Трекинг экспериментов (ClearML)](docs/tracking.ru.md)
- [Справочник по API](docs/api-reference.ru.md) — конструктор `Evaluation`, конфиги, атрибуты

**Справочные материалы**

- [Определения метрик](docs/metrics-definitions.ru.md) — IoU/TP/FP/FN, mAP, методы AP
- [Почему P/R/F1 расходится с mAP](docs/why_prf1_differs.md) — с графиками (на английском)
- [Changelog](CHANGELOG.md)

---

## Разработка

```bash
git clone https://github.com/Wasilkas/digital-metrics
cd digital-metrics
uv venv && uv sync

uv run ruff check . --fix
uv run ruff format .
uv run mypy src/
uv run pytest --cov=src/digital_metrics tests/
```
