"""Realizable IoU-0.50 operating points, separate from backend AP envelopes."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

from ..matching import compute_iou_matrix
from ..types import DetectionMetrics
from ..validation import (
    BBOX_COLS,
    drop_na_labels,
    normalize_image_ids,
    validate_dataframes,
    validate_option,
)


def prepare_inputs(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    classes: list[str] | None,
    images: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    validate_dataframes(preds, gt)
    if images is not None:
        gt, preds, scope_frame = normalize_image_ids(
            gt, preds, pd.DataFrame({"image_name": images})
        )
        images = scope_frame["image_name"].tolist()
    else:
        gt, preds = normalize_image_ids(gt, preds)
    scope = set(gt["image_name"]) | ({str(i) for i in images} if images is not None else set())
    preds = drop_na_labels(preds)
    if classes is None:
        classes = sorted(set(gt["instance_label"].dropna()) | set(preds["instance_label"]))
    elif "background" in classes:
        raise ValueError("'background' is reserved for unmatched detections.")
    unknown = preds.loc[~preds["instance_label"].isin(classes), "instance_label"]
    if not unknown.empty:
        logger.warning(
            f"Dropping predictions outside explicit classes: {unknown.value_counts().to_dict()}"
        )
    preds = preds[preds["instance_label"].isin(classes) & preds["image_name"].isin(scope)]
    gt = gt[gt["instance_label"].isin(classes) | gt["instance_label"].isna()]
    return gt, preds, classes


@dataclass
class _ImageGroup:
    names: list[str]
    pred_classes: npt.NDArray[np.int64]
    gt_classes: npt.NDArray[np.int64]
    scores: npt.NDArray[np.float64]
    order: npt.NDArray[np.intp]
    iou: npt.NDArray[np.float64]


def _coco_corners(boxes: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Mirror tensor xyxy -> float32 xywh -> COCO double-precision corners."""
    sizes = boxes[:, 2:] - boxes[:, :2]
    corners = boxes.astype(np.float64)
    corners[:, 2:] = corners[:, :2] + sizes.astype(np.float64)
    return corners


def _prepare_groups(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    backend: Literal["ultralytics", "torchmetrics"],
    classes: list[str],
) -> list[_ImageGroup]:
    """Cache COCO image/classes or complete Ultralytics images in row order.

    Ultralytics' whole-image IoU tie sort can depend on other classes even
    though cross-class matches are masked out, so its groups must stay intact.
    """
    keys = ["image_name", "instance_label"] if backend == "torchmetrics" else ["image_name"]
    labels = {name: i for i, name in enumerate(classes)}
    targets = {
        key: group for key, group in gt.dropna(subset=["instance_label"]).groupby(keys, sort=False)
    }
    groups = []
    empty = gt.iloc[:0]
    for key, group in preds.groupby(keys, sort=False):
        boxes = group[BBOX_COLS].to_numpy(np.float32)
        target_df = targets.get(key, empty)
        target = target_df[BBOX_COLS].to_numpy(np.float32)
        if backend == "torchmetrics":
            iou = compute_iou_matrix(_coco_corners(boxes), _coco_corners(target))
        else:
            import torch
            from ultralytics.utils.metrics import box_iou

            iou = box_iou(torch.tensor(target), torch.tensor(boxes)).cpu().numpy().T
        scores = group["confidence"].to_numpy(np.float64)
        # The AP adapter sends float32 scores to COCO. Quantization can create
        # ties; their stable order must remain the original DataFrame order.
        order = np.argsort(-scores.astype(np.float32), kind="stable")
        groups.append(
            _ImageGroup(
                classes,
                np.asarray(group["instance_label"].map(labels), dtype=np.int64),
                np.asarray(target_df["instance_label"].map(labels), dtype=np.int64),
                scores,
                order,
                iou,
            )
        )
    return groups


def _group_counts(
    group: _ImageGroup,
    threshold: float | dict[str, float],
    backend: Literal["ultralytics", "torchmetrics"],
) -> dict[str, tuple[int, int]]:
    """Re-match one changed group, retaining the original mixed-class row order."""
    thresholds = (
        np.array([threshold.get(group.names[int(c)], 0.0) for c in group.pred_classes])
        if isinstance(threshold, dict)
        else threshold
    )
    retained = group.scores >= thresholds
    if backend == "torchmetrics":
        positions = group.order[retained[group.order]][:100]
        used: set[int] = set()
        for row in group.iou[positions]:
            best, best_iou = -1, 0.5
            for j, overlap in enumerate(row):
                if j not in used and overlap >= best_iou:
                    best, best_iou = j, float(overlap)
            if best != -1:
                used.add(best)
        return {group.names[int(group.pred_classes[0])]: (len(used), len(positions))}
    from .ultralytics_metrics import _match_predictions

    iou = group.iou[retained]
    correct = _match_predictions(
        group.pred_classes[retained],
        group.gt_classes,
        iou.T,
    )
    retained_classes = group.pred_classes[retained]
    return {
        group.names[int(c)]: (
            int(correct[retained_classes == c, 0].sum()),
            int((retained_classes == c).sum()),
        )
        for c in np.unique(group.pred_classes)
    }


def raw_counts(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    classes: list[str] | None,
    images: list[str] | None,
    threshold: float | dict[str, float],
    backend: Literal["ultralytics", "torchmetrics"],
) -> dict[str, tuple[int, int, int]]:
    """Retain original confidence >= threshold and apply backend matching.

    COCO boxes and stable score ordering use the same float32 inputs as the AP
    adapter, with its 100 detections per image/class limit. Ultralytics retains
    its versioned whole-image validator assignment and mixed-class row order,
    without an imposed COCO cap.
    """
    gt, preds, classes = prepare_inputs(gt, preds, classes, images)
    thresholds = {
        name: threshold.get(name, 0.0) if isinstance(threshold, dict) else threshold
        for name in classes
    }
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in thresholds.values()):
        raise ValueError("conf_threshold must be finite and in [0, 1].")
    totals = {name: [0, 0] for name in classes}
    for group in _prepare_groups(gt, preds, backend, classes):
        for name, (tp, count) in _group_counts(group, thresholds, backend).items():
            totals[name][0] += tp
            totals[name][1] += count
    positives = gt["instance_label"].value_counts()
    return {
        name: (tp, count - tp, int(positives.get(name, 0)) - tp)
        for name, (tp, count) in totals.items()
    }


def raw_prf1(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    classes: list[str] | None,
    images: list[str] | None,
    threshold: float | dict[str, float],
    backend: Literal["ultralytics", "torchmetrics"],
) -> dict[str, tuple[float, float, float]]:
    result = {}
    for name, (tp, fp, fn) in raw_counts(gt, preds, classes, images, threshold, backend).items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        result[name] = p, r, 2 * p * r / (p + r) if p + r else 0.0
    return result


class _DictionaryCounts:
    """Cached group counts and the exact sum of realized class F1 values."""

    def __init__(self, positives: dict[str, int]) -> None:
        self.positives = positives
        self.totals = {name: [0, 0] for name in positives}
        self.groups: dict[int, dict[str, tuple[int, int]]] = {}
        self.values = {name: Fraction(0) for name in positives}
        self.objective = Fraction(0)

    def copy(self) -> "_DictionaryCounts":
        other = _DictionaryCounts(self.positives)
        other.totals = {name: counts.copy() for name, counts in self.totals.items()}
        other.groups = self.groups.copy()
        other.values = self.values.copy()
        other.objective = self.objective
        return other

    def update(self, i: int, counts: dict[str, tuple[int, int]]) -> None:
        previous = self.groups.get(i, {})
        for name, (tp, count) in counts.items():
            if name not in self.totals:
                continue
            previous_tp, previous_count = previous.get(name, (0, 0))
            self.totals[name][0] += tp - previous_tp
            self.totals[name][1] += count - previous_count
            total_tp, total_count = self.totals[name]
            value = Fraction(2 * total_tp, self.positives[name] + total_count)
            self.objective += value - self.values[name]
            self.values[name] = value
        self.groups[i] = counts


def _dictionary_counts(
    groups: list[_ImageGroup], thresholds: dict[str, float], positives: dict[str, int]
) -> _DictionaryCounts:
    state = _DictionaryCounts(positives)
    for i, group in enumerate(groups):
        state.update(i, _group_counts(group, thresholds, "ultralytics"))
    return state


def _refine_ultralytics_dictionary(
    groups: list[_ImageGroup],
    thresholds: dict[str, float],
    positives: dict[str, int],
    global_threshold: float,
    highest_score: float,
) -> dict[str, float]:
    """Coordinate-ascent local macro-F1 optimum for coupled whole-image ties.

    Each sweep fixes the other thresholds and rematches only image groups whose
    current class retention changes. Strict objective gains or equal-objective
    threshold increases guarantee termination on the finite candidate grid.
    """
    if not any(len(np.unique(group.pred_classes)) > 1 for group in groups):
        return thresholds
    state = _dictionary_counts(groups, thresholds, positives)
    scalar_seed = {name: global_threshold for name in thresholds}
    scalar_state = _dictionary_counts(groups, scalar_seed, positives)
    # A scalar seed is also evaluated as the actual returned dictionary, so
    # absent-GT classes retain their public default threshold of zero.
    names = list(thresholds)
    if scalar_state.objective > state.objective or (
        scalar_state.objective == state.objective
        and tuple(scalar_seed[name] for name in names) > tuple(thresholds[name] for name in names)
    ):
        thresholds, state = scalar_seed, scalar_state
    events: dict[str, dict[float, list[int]]] = {name: {} for name in names}
    affected: dict[str, list[int]] = {name: [] for name in names}
    for i, group in enumerate(groups):
        for c in np.unique(group.pred_classes):
            name = group.names[int(c)]
            if name not in events:
                continue
            affected[name].append(i)
            for score in np.unique(group.scores[group.pred_classes == c]):
                events[name].setdefault(float(score), []).append(i)
    improved = True
    while improved:
        improved = False
        for name in names:
            candidates = sorted(set(events[name]) | {highest_score}, reverse=True)
            working = state.copy()
            proposal = thresholds | {name: candidates[0]}
            for i in affected[name]:
                working.update(i, _group_counts(groups[i], proposal, "ultralytics"))
            best_objective, best_threshold = working.objective, candidates[0]
            for candidate in candidates[1:]:
                proposal[name] = candidate
                for i in events[name].get(candidate, []):
                    working.update(i, _group_counts(groups[i], proposal, "ultralytics"))
                if working.objective > best_objective:
                    best_objective, best_threshold = working.objective, candidate
            if best_objective > state.objective or (
                best_objective == state.objective and best_threshold > thresholds[name]
            ):
                thresholds[name] = best_threshold
                for i in affected[name]:
                    state.update(i, _group_counts(groups[i], thresholds, "ultralytics"))
                improved = True
    return thresholds


def calibrate_raw(
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    classes: list[str] | None,
    images: list[str] | None,
    mode: str,
    backend: Literal["ultralytics", "torchmetrics"],
) -> float | dict[str, float]:
    validate_option("mode", mode, ("global", "per_class"))
    gt, preds, classes = prepare_inputs(gt, preds, classes, images)
    positives = gt["instance_label"].value_counts().to_dict()
    present = [name for name in classes if positives.get(name, 0) > 0]
    groups = _prepare_groups(gt, preds, backend, classes)
    # A score event changes only groups containing that original confidence.
    # Ultralytics rematches the full changed image and updates every class
    # affected by its IoU tie ordering. COCO updates only the image/class.
    events: dict[float, list[int]] = {}
    for i, group in enumerate(groups):
        for score in np.unique(group.scores):
            events.setdefault(float(score), []).append(i)
    if not events or not present:
        return 0.0 if mode == "global" else {name: 0.0 for name in present}
    candidates = sorted(events, reverse=True)
    totals = {name: [0, 0] for name in present}
    group_totals: list[dict[str, tuple[int, int]]] = [{} for _ in groups]
    values = {name: Fraction(0) for name in present}
    best = {name: (Fraction(0), candidates[0]) for name in present}
    global_sum = Fraction(0)
    global_best = Fraction(-1)
    global_threshold = candidates[0]
    for threshold in candidates:
        changed: set[str] = set()
        for i in events[threshold]:
            group = groups[i]
            counts = _group_counts(group, threshold, backend)
            for name, (tp, count) in counts.items():
                if name not in totals:
                    continue
                previous_tp, previous_count = group_totals[i].get(name, (0, 0))
                totals[name][0] += tp - previous_tp
                totals[name][1] += count - previous_count
                changed.add(name)
            group_totals[i] = counts
        for name in changed:
            tp, count = totals[name]
            # Exact objective comparisons preserve the highest equal optimum,
            # including ties involving multiple classes and incremental updates.
            value = Fraction(2 * tp, int(positives[name]) + count)
            global_sum += value - values[name]
            values[name] = value
            if value > best[name][0]:
                best[name] = value, threshold
        if global_sum > global_best:
            global_best, global_threshold = global_sum, threshold
    if mode == "global":
        return global_threshold
    thresholds = {name: best[name][1] for name in present}
    if backend == "ultralytics":
        return _refine_ultralytics_dictionary(
            groups,
            thresholds,
            {name: int(positives[name]) for name in present},
            global_threshold,
            candidates[0],
        )
    return thresholds


def complete_counts(
    out: dict[str, "DetectionMetrics"],
    gt: pd.DataFrame,
    preds: pd.DataFrame,
    classes: list[str] | None,
    images: list[str] | None,
    threshold: float | dict[str, float] | None,
    backend: Literal["ultralytics", "torchmetrics"],
) -> dict[str, "DetectionMetrics"]:
    """Attach observed counts, including classes absent from the GT split."""
    if threshold is None:
        return out
    for name, (tp, fp, fn) in raw_counts(gt, preds, classes, images, threshold, backend).items():
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        value = out.get(name)
        if value is None:
            value = DetectionMetrics(
                precision=p,
                recall=r,
                f1=0,
                ap50=float("nan"),
                ap75=float("nan"),
                ap50_95=float("nan"),
            )
            out[name] = value
        value.tp, value.fp, value.fn = tp, fp, fn
        value.precision, value.recall = p, r
        value.f1 = 2 * p * r / (p + r) if p + r else 0.0
    return out
