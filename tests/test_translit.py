"""Transliterated label restoration: schemes, index, matching, Evaluation wiring."""

import pandas as pd
import pytest

from digital_metrics import Evaluation
from digital_metrics.translit import (
    build_translit_index,
    normalize_label,
    restore_labels,
    transliterate,
)

GT_LABELS = ["Грязь на основе", "ВкатЛО", "Дефект Кромки", "ЦарапинаМатовая", "Щётка"]


@pytest.mark.parametrize(
    ("scheme", "expected"),
    [
        ("common", "Gryaz na osnove"),
        ("gost", "Gryaz na osnove"),
        ("icao", "Griaz na osnove"),
    ],
)
def test_transliterate_schemes(scheme: str, expected: str) -> None:
    assert transliterate("Грязь на основе", scheme) == expected


def test_transliterate_preserves_case_and_non_cyrillic() -> None:
    assert transliterate("ВкатЛО") == "VkatLO"
    assert transliterate("Наплыв 1-2кат") == "Naplyv 1-2kat"


def test_transliterate_unknown_scheme() -> None:
    with pytest.raises(ValueError, match="Unknown transliteration scheme"):
        transliterate("Грязь", "klingon")


def test_normalize_label_drops_separators_and_case() -> None:
    assert normalize_label("Gryaz_na-osnove") == normalize_label("gryaz na osnove")
    assert normalize_label("GryazNaOsnove") == "gryaznaosnove"


def test_index_covers_every_scheme() -> None:
    index = build_translit_index(GT_LABELS)
    for key in ("gryaznaosnove", "griaznaosnove", "vkatlo", "shchetka", "shhetka", "shhyotka"):
        assert key in index


@pytest.mark.parametrize(
    ("model_label", "expected"),
    [
        ("Gryaz_na_osnove", "Грязь на основе"),  # common scheme, underscores
        ("griaz_na_osnove", "Грязь на основе"),  # ICAO scheme, lower case
        ("VkatLO", "ВкатЛО"),  # CamelCase, no separators
        ("Defekt Kromki", "Дефект Кромки"),  # spaces preserved
        ("shhetka", "Щётка"),  # GOST щ → shh, ё → e
        ("Tsarapina_Matovaya", "ЦарапинаМатовая"),  # extra separator vs GT
        ("Carapina-matovaya", "ЦарапинаМатовая"),  # GOST ц → c
        ("Carapina_Matovaia", "ЦарапинаМатовая"),  # mixed schemes → fuzzy fallback
        ("Грязь на основе", "Грязь на основе"),  # already a GT label
    ],
)
def test_restore_labels_matches(model_label: str, expected: str) -> None:
    assert restore_labels([model_label], GT_LABELS)[model_label] == expected


def test_restore_labels_warns_and_keeps_unmatched() -> None:
    mapping = restore_labels(["Gryaz_na_osnove", "totally_other_class"], GT_LABELS)
    assert mapping["Gryaz_na_osnove"] == "Грязь на основе"
    assert mapping["totally_other_class"] == "totally_other_class"


def test_restore_labels_cutoff_is_enforced() -> None:
    # A spelling no scheme produces exactly ("c" for ц with "ia" for я) resolves
    # through the fuzzy fallback at the default cutoff, but not at 1.0.
    label = "Carapina_Matovaia"
    assert restore_labels([label], GT_LABELS)[label] == "ЦарапинаМатовая"
    assert restore_labels([label], GT_LABELS, cutoff=1.0)[label] == label


def test_restore_labels_rejects_bad_cutoff() -> None:
    with pytest.raises(ValueError, match=r"cutoff must be in \(0, 1\]"):
        restore_labels(["x"], GT_LABELS, cutoff=0.0)


def test_ambiguous_key_is_not_restored() -> None:
    # Two GT classes whose transliterations collide: neither claims the key.
    mapping = restore_labels(["kromka"], ["Кромка", "КРОМКА"], cutoff=1.0)
    assert mapping["kromka"] == "kromka"


def _cyrillic_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_df = pd.DataFrame(
        [
            ("img1", "Грязь на основе", 0, 0, 100, 100, "test"),
            ("img1", "ВкатЛО", 200, 200, 300, 300, "test"),
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "split",
        ],
    )
    preds_df = pd.DataFrame(
        [
            ("img1", "Gryaz_na_osnove", 0, 0, 100, 100, 0.9),
            ("img1", "VkatLO", 200, 200, 300, 300, 0.8),
        ],
        columns=[
            "image_name",
            "instance_label",
            "bbox_x_tl",
            "bbox_y_tl",
            "bbox_x_br",
            "bbox_y_br",
            "confidence",
        ],
    )
    return gt_df, preds_df


def test_evaluation_without_flag_scores_zero() -> None:
    gt_df, preds_df = _cyrillic_dataset()
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5)
    evaluation("test")
    assert all(m.tp == 0 for m in evaluation.metrics.values())


def test_evaluation_with_transliterated_labels_scores_perfect() -> None:
    gt_df, preds_df = _cyrillic_dataset()
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5, transliterated_labels=True)
    evaluation("test")
    assert {label: m.tp for label, m in evaluation.metrics.items()} == {
        "Грязь на основе": 1,
        "ВкатЛО": 1,
    }
    assert set(evaluation.preds_df["instance_label"]) == {"Грязь на основе", "ВкатЛО"}
    assert set(evaluation._raw_preds_df["instance_label"]) == {"Грязь на основе", "ВкатЛО"}


def test_evaluation_unmatched_label_is_left_and_dropped() -> None:
    gt_df, preds_df = _cyrillic_dataset()
    preds_df.loc[1, "instance_label"] = "some_other_thing"
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5, transliterated_labels=True)
    evaluation("test")
    # Restoration left it alone; the unknown-class guard then dropped the row.
    assert set(evaluation.preds_df["instance_label"]) == {"Грязь на основе"}
    assert evaluation.metrics["ВкатЛО"].fn == 1


def test_restoration_is_idempotent() -> None:
    gt_df, preds_df = _cyrillic_dataset()
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5, transliterated_labels=True)
    evaluation("test")
    first = evaluation.preds_df["instance_label"].tolist()
    evaluation("test")
    assert evaluation.preds_df["instance_label"].tolist() == first


def test_restore_labels_ignores_non_string_gt_labels() -> None:
    # Empty images carry a placeholder GT row with a NaN label.
    mapping = restore_labels(["Gryaz_na_osnove", float("nan")], [*GT_LABELS, float("nan")])
    assert mapping == {"Gryaz_na_osnove": "Грязь на основе"}


def test_evaluation_with_empty_image_gt_rows() -> None:
    gt_df, preds_df = _cyrillic_dataset()
    empty_image = pd.DataFrame(
        [("img2", None, None, None, None, None, "test")], columns=gt_df.columns
    )
    gt_df = pd.concat([gt_df, empty_image], ignore_index=True)
    evaluation = Evaluation(preds_df, gt_df, iou_threshold=0.5, transliterated_labels=True)
    evaluation("test")
    assert evaluation.classes == ["Грязь на основе", "ВкатЛО"]
    assert {label: m.tp for label, m in evaluation.metrics.items()} == {
        "Грязь на основе": 1,
        "ВкатЛО": 1,
    }
