"""Recovering Cyrillic class labels from transliterated model outputs.

Datasets are often converted to a training format that transliterates the class
names (``"Грязь на основе"`` → ``"Gryaz_na_osnove"``), so a model trained on
them predicts Latin labels while the ground truth stays Cyrillic. Every
prediction would then be an unknown class and every metric would be zero.

Reverse transliteration is ambiguous (``kh``/``h``, ``ya``/``ia``/``ja`` all map
back to the same Cyrillic letter), so instead of inverting the transformation we
go forward: each ground-truth label is transliterated with several common
schemes, the results are normalised (lower-cased, non-alphanumerics dropped) and
indexed. A model label is normalised the same way and looked up in that index,
falling back to fuzzy matching. Labels with no close match are reported in a
warning and left untouched.

A tiny foundation module — stdlib + loguru only.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

from loguru import logger

# Characters shared by every scheme; only the ambiguous letters differ below.
_BASE: dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "з": "z",
    "и": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "ч": "ch",
    "ш": "sh",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ж": "zh",
}

# The three transliteration conventions seen in practice. They agree on most
# letters and differ exactly where converters disagree, so indexing all three
# covers the common cases; fuzzy matching absorbs the rest.
TRANSLIT_SCHEMES: dict[str, dict[str, str]] = {
    # Yandex/Wikipedia style: kh / ts / shch / yu / ya, ё → e.
    "common": _BASE | {"ё": "e", "й": "y", "х": "kh", "ц": "ts", "щ": "shch", "ю": "yu", "я": "ya"},
    # ISO 9 / GOST 7.79 System B flavour: h / c / shh, j for й, yo for ё.
    "gost": _BASE | {"ё": "yo", "й": "j", "х": "h", "ц": "c", "щ": "shh", "ю": "yu", "я": "ya"},
    # ICAO / passport style: i for й, iu / ia endings.
    "icao": _BASE | {"ё": "e", "й": "i", "х": "kh", "ц": "ts", "щ": "shch", "ю": "iu", "я": "ia"},
}

DEFAULT_SCHEME = "common"


def transliterate(text: str, scheme: str = DEFAULT_SCHEME) -> str:
    """Transliterate Cyrillic ``text`` to Latin using ``scheme``.

    Args:
        text: Source string; non-Cyrillic characters pass through unchanged.
        scheme: One of :data:`TRANSLIT_SCHEMES` (``"common"``, ``"gost"``,
            ``"icao"``). Defaults to ``"common"``.

    Returns:
        The transliterated string, preserving the case of the source letters
        (an upper-case Cyrillic letter yields a capitalised Latin group).

    Raises:
        ValueError: If ``scheme`` is unknown.
    """
    try:
        table = TRANSLIT_SCHEMES[scheme]
    except KeyError:
        raise ValueError(
            f"Unknown transliteration scheme {scheme!r}; expected one of "
            f"{sorted(TRANSLIT_SCHEMES)}."
        ) from None

    out: list[str] = []
    for char in text:
        latin = table.get(char.lower())
        if latin is None:
            out.append(char)
        elif char.isupper():
            out.append(latin.capitalize())
        else:
            out.append(latin)
    return "".join(out)


def normalize_label(text: str) -> str:
    """Fold a label to its comparison key: lower-cased, alphanumerics only.

    Drops the separators converters disagree about (spaces, underscores,
    hyphens, dots), so ``"Gryaz na osnove"``, ``"gryaz_na_osnove"`` and
    ``"GryazNaOsnove"`` share one key.
    """
    return "".join(char for char in text.lower() if char.isalnum())


def _strings(labels: Iterable[object]) -> list[str]:
    """Keep only the string labels; ``NaN``/``None`` placeholders are not classes."""
    return [label for label in labels if isinstance(label, str)]


def build_translit_index(gt_labels: Iterable[str]) -> dict[str, str]:
    """Index the ground-truth vocabulary by every normalised spelling of it.

    Each label contributes its own normalised form plus one per scheme in
    :data:`TRANSLIT_SCHEMES`. Keys claimed by two different labels are dropped
    (with a warning) rather than resolved arbitrarily. Non-string entries (the
    ``NaN`` label of an empty-image placeholder row) are skipped.

    Args:
        gt_labels: Ground-truth class names.

    Returns:
        Mapping of normalised key → ground-truth label.
    """
    index: dict[str, str] = {}
    ambiguous: set[str] = set()
    for label in _strings(gt_labels):
        # ``ё``/``е`` are used interchangeably in practice, on both sides of the
        # conversion, so index the folded spelling as well.
        spellings = {label, label.replace("ё", "е").replace("Ё", "Е")}
        keys = {normalize_label(label)} | {
            normalize_label(transliterate(spelling, scheme))
            for spelling in spellings
            for scheme in TRANSLIT_SCHEMES
        }
        for key in keys:
            if not key or key in ambiguous:
                continue
            owner = index.get(key)
            if owner is not None and owner != label:
                logger.warning(
                    f"Transliterated spelling {key!r} is shared by ground-truth classes "
                    f"{owner!r} and {label!r}; ignoring it for label restoration."
                )
                del index[key]
                ambiguous.add(key)
                continue
            index[key] = label
    return index


def restore_labels(
    labels: Iterable[str],
    gt_labels: Iterable[str],
    *,
    cutoff: float = 0.8,
) -> dict[str, str]:
    """Map transliterated model labels back onto the ground-truth vocabulary.

    A label already present in the ground truth is kept as is. Otherwise its
    normalised form is looked up among the normalised transliterations of the
    ground-truth labels, then — if that misses — matched fuzzily against the same
    keys. Labels with no match at or above ``cutoff`` are reported in a single
    warning and map to themselves. Non-string entries on either side (the
    ``NaN`` label of an empty-image placeholder row) are ignored.

    Args:
        labels: Model / prediction class names to restore.
        gt_labels: Ground-truth class vocabulary to restore them onto.
        cutoff: Minimum :mod:`difflib` similarity (0–1] for a fuzzy match.
            Higher is stricter; 1.0 disables fuzzy matching in practice.

    Returns:
        Mapping of every input label → restored label (identity when unresolved).

    Raises:
        ValueError: If ``cutoff`` is outside ``(0, 1]``.
    """
    if not 0 < cutoff <= 1:
        raise ValueError(f"cutoff must be in (0, 1], got {cutoff}.")

    known = list(dict.fromkeys(_strings(gt_labels)))
    known_set = set(known)
    index = build_translit_index(known)
    keys = list(index)

    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for label in dict.fromkeys(_strings(labels)):
        if label in known_set:
            mapping[label] = label
            continue
        key = normalize_label(label)
        target = index.get(key)
        if target is None and key:
            close = difflib.get_close_matches(key, keys, n=1, cutoff=cutoff)
            target = index[close[0]] if close else None
        if target is None:
            unresolved.append(label)
            mapping[label] = label
        else:
            mapping[label] = target

    restored = {src: dst for src, dst in mapping.items() if src != dst}
    if restored:
        summary = ", ".join(f"{src!r} -> {dst!r}" for src, dst in sorted(restored.items()))
        logger.info(f"Restored {len(restored)} transliterated label(s): {summary}.")
    if unresolved:
        logger.warning(
            f"No ground-truth class close enough (cutoff={cutoff}) for "
            f"{len(unresolved)} label(s): {sorted(unresolved)}; keeping them unchanged."
        )
    return mapping
