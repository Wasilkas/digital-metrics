[← Оглавление документации](README.ru.md) · [🇬🇧 English version](outputs.md)

# Результаты, дашборды и аудит ошибок

## `ev.metrics` — `dict[str, Metrics]`

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

## Дашборды и графики

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

## Аудит ошибок

```python
# Top-k пар предсказание/эталон, перепутанных между двумя классами
audit_df = ev.get_topk_confusions(main_class="car", k=20)

# DataFrame с разметкой типа сопоставления для визуализации.
# gt_df:    predict_type ∈ {"TP", "FN"}
# preds_df: predict_type ∈ {"TP", "FP"}
gt_vis, pred_vis = ev.get_dfs_visualization()

# apply_thresholds=True применяет пороги best_confidences по классам (как
# slice_by_conf): предсказания ниже порога своего класса становятся "filtered",
# а эталон, обнаруженный только таким отфильтрованным предсказанием, — "FN".
gt_vis, pred_vis = ev.get_dfs_visualization(apply_thresholds=True)
```

