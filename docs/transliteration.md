[← Documentation index](README.md) · [🇷🇺 Русская версия](transliteration.ru.md)

# Transliterated model labels

Dataset converters often transliterate class names before training
(`"Грязь на основе"` → `"Gryaz_na_osnove"`), so the model predicts Latin labels
while the ground truth stays Cyrillic. Every prediction is then an unknown class
and every metric comes out zero. Pass `transliterated_labels=True` to restore
them:

```python
ev = Evaluation(preds_df, gt_df, transliterated_labels=True)
ev(split="test")
```

Reverse transliteration is ambiguous (`kh`/`h`, `ya`/`ia`/`ja` all come from one
Cyrillic letter), so the restoration runs **forward** off the ground truth: each
GT label is transliterated with three common schemes (`common` Yandex-style,
`gost` ISO 9 / GOST 7.79-B, `icao` passport-style, plus a `ё`→`е` fold),
normalised — lower-cased with spaces, underscores, hyphens and dots dropped — and
indexed. Each prediction label is normalised the same way and looked up:

1. already a GT label → kept as is;
2. exact key hit → restored (case, separators and scheme differences absorbed);
3. otherwise a fuzzy match against the same keys at `translit_match_cutoff`
   (default `0.8`, `difflib` similarity; raise it to be stricter);
4. no close match → a **warning** listing those labels, and they are left
   unchanged (so they are then dropped as unknown classes).

Restoration rewrites the labels in both `preds_df` and `_raw_preds_df` before
scoring, so the native and both external backends see the GT vocabulary. A
transliterated spelling shared by two different GT classes is reported and left
unmatched rather than resolved arbitrarily. The helpers are public:

```python
from digital_metrics import transliterate, restore_labels

transliterate("Грязь на основе")                 # 'Gryaz na osnove'
transliterate("Грязь на основе", "icao")         # 'Griaz na osnove'
restore_labels(["Gryaz_na_osnove"], gt_labels)   # {'Gryaz_na_osnove': 'Грязь на основе'}
```

