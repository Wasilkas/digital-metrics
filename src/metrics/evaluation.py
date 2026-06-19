from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from .backends import (
    Backend,
    compute_detection_metrics,
    compute_ultralytics_confusion_matrix,
    compute_ultralytics_metrics,
    find_ultralytics_confidence,
)
from .inference import ImageNameMode, predict_on_images
from .matching import (
    MatchingStrategy,
    compute_iou_matrix,
    find_duplicates_bboxes,
    match_boxes,
)
from .preprocess import apply_nms, filter_by_confidence
from .reporting import get_dashboards, plot_confidence_intervals
from .scoring import (
    APMethod,
    ConfidenceOptimization,
    compute_kappa,
    compute_map,
    find_best_confidences,
    find_best_global_confidence,
    get_confusion_matrix,
    get_confusions,
    slice_by_conf,
)
from .types import DetectionMetrics, Metrics, PredictMatch

_REQUIRED_COLS_GT = {
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
}
_REQUIRED_COLS_PREDS = {
    "image_name",
    "instance_label",
    "bbox_x_tl",
    "bbox_y_tl",
    "bbox_x_br",
    "bbox_y_br",
    "confidence",
}


def _compute_metrics_from_matches(
    matches: dict[str, list[PredictMatch]],
    classes: list[str],
    best_confidences: dict[str, float],
) -> dict[str, Metrics]:
    result: dict[str, Metrics] = {}
    for c in classes:
        m = Metrics()
        m.confidence = best_confidences.get(c, 0.0)
        for match in matches.get(c, []):
            pred_type = match.type.lower()
            setattr(m, pred_type, getattr(m, pred_type) + 1)
        result[c] = m
    return result


class Evaluation:
    """Orchestrator for object-detection evaluation metrics."""

    def __init__(
        self,
        preds_df: pd.DataFrame | str | None,
        split_df: pd.DataFrame | str,
        iou_threshold: float = 0.5,
        *,
        preprocess: bool = False,
        skip_cohen_kappa: bool = True,
        matching_strategy: MatchingStrategy = "iou_prior",
        preprocess_preds_conf_threshold: float | None = None,
        preprocess_preds_nms_containment_threshold: float | None = None,
        preprocess_preds_nms_iou_threshold: float | None = None,
        ap_method: APMethod = "interp",
        confidence_optimization: ConfidenceOptimization = "per_class",
        weights_path: str | None = None,
        backend: Backend | None = None,
        predict_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the Evaluation object.

        Args:
            preds_df: Predictions DataFrame or path to CSV. Pass ``None`` to start
                without predictions and generate them from a YOLO model via
                :meth:`predict_to_dataframe` (which reads image paths from
                ``split_df['image_path']``).
            split_df: Ground-truth split DataFrame or path to CSV.
            iou_threshold: IoU threshold for box matching. Defaults to 0.5.
            preprocess: Remove duplicate GT boxes if True. Defaults to False.
            skip_cohen_kappa: Skip cohen_kappa computation. Defaults to True.
            matching_strategy: "iou_prior" (default, Ultralytics non-scipy
                style), "greedy" (YOLO confidence-sorted), or "hungarian".
            preprocess_preds_conf_threshold: Drop predictions whose confidence
                is strictly below this value before evaluation.  None disables
                confidence filtering.
            preprocess_preds_nms_containment_threshold: Same-class containment
                threshold for custom NMS.  For each pair of same-class boxes,
                the lower-confidence one is suppressed when
                intersection / min(area_a, area_b) >= threshold (i.e. one box
                is largely inside the other).  None disables same-class
                containment suppression (equivalent to setting threshold > 1).
            preprocess_preds_nms_iou_threshold: Cross-class IoU threshold for
                custom NMS.  The lower-confidence prediction is suppressed when
                two different-class boxes have IoU >= threshold.  None disables
                cross-class NMS.
            ap_method: AP integration method for mAP computation — ``"interp"``
                (default, 101-point COCO interpolation, Ultralytics-compatible)
                or ``"continuous"`` (VOC 2010+ rectangle-area).
            confidence_optimization: How to choose confidence thresholds when
                ``find_best_confs`` is enabled or a ``calibration_split`` is
                used. ``"per_class"`` (default) tunes a separate threshold per
                class to maximise that class's F1. ``"global"`` mirrors YOLO and
                picks a single threshold, shared by every class, that maximises
                the mean per-class F1.
            weights_path: Optional path to YOLO weights. When ``preds_df`` is
                ``None``, predictions are generated from these weights the first
                time the evaluation runs (over ``split_df['image_path']``). If
                ``preds_df`` is ``None`` and no weights are given, calling the
                evaluation raises ``ValueError``.
            backend: Which metrics engine to use. ``None`` (default) runs the
                native pipeline (custom matching, confusion matrix, Cohen's kappa,
                confidence calibration). ``"ultralytics"`` or ``"torchmetrics"``
                instead score the split with that external library (over the raw,
                unpreprocessed predictions, the way YOLO val does) and expose the
                results both as ``detection_metrics`` (per-class
                :class:`DetectionMetrics`) and as native ``metrics`` adapted so the
                dashboards keep working. By default a backend self-selects its
                operating point on the eval split; pass ``calibration_split`` to
                instead read P/R/F1 at the F1-optimal confidence found on that
                split (``"ultralytics"`` only — AP stays over the full curve;
                ``"torchmetrics"`` ignores it with a warning). ``find_best_confs``
                does not apply in backend mode.
            predict_kwargs: Extra keyword arguments forwarded to Ultralytics'
                ``model.predict`` when predictions are auto-generated from
                ``weights_path`` (e.g. ``{"conf": 0.25, "imgsz": 1280, "half":
                True, "augment": True}``). Ignored when ``preds_df`` is provided.
                For one-off control you can instead call
                :meth:`predict_to_dataframe` with the same keyword arguments.
        """
        self.suffix = "default"

        if isinstance(preds_df, str):
            preds_df = pd.read_csv(preds_df)
        if isinstance(split_df, str):
            split_df = pd.read_csv(split_df)

        self._weights_path = weights_path
        self._has_predictions = preds_df is not None
        if not self._has_predictions and weights_path is not None:
            logger.info(
                f"No predictions provided; they will be generated from weights "
                f"'{weights_path}' when the evaluation runs."
            )
        if self._has_predictions and weights_path is not None:
            logger.warning(
                "Both preds_df and weights_path were provided; using preds_df and "
                "ignoring weights_path."
            )
        if preds_df is None:
            # Placeholder until predictions are generated (predict_to_dataframe).
            preds_df = pd.DataFrame(columns=sorted(_REQUIRED_COLS_PREDS))

        self.preds_df: pd.DataFrame = preds_df.reset_index(drop=True)
        self._raw_preds_df: pd.DataFrame = self.preds_df.copy()
        self.split_df: pd.DataFrame = split_df.reset_index(drop=True)
        self.gt_df: pd.DataFrame | None = None

        self.iou_threshold = iou_threshold
        # Defer KeyError: _validate_df will raise ValueError with a clear message if missing.
        self.classes: list[str] = (
            self.split_df["instance_label"].unique().tolist()
            if "instance_label" in self.split_df.columns
            else []
        )
        self._best_confidences: dict[str, float] = {c: 0.0 for c in self.classes}
        self._skip_cohen_kappa = skip_cohen_kappa
        self.matching_strategy: MatchingStrategy = matching_strategy
        self._preds_conf_threshold: float | None = preprocess_preds_conf_threshold
        self._preds_nms_containment_threshold: float | None = (
            preprocess_preds_nms_containment_threshold
        )
        self._preds_nms_iou_threshold: float | None = preprocess_preds_nms_iou_threshold
        self._ap_method: APMethod = ap_method
        self._confidence_optimization: ConfidenceOptimization = confidence_optimization
        self._backend: Backend | None = backend
        self._predict_kwargs: dict[str, Any] = predict_kwargs or {}

        self.metrics: dict[str, Metrics] = {}
        self.detection_metrics: dict[str, DetectionMetrics] = {}
        self.cm: npt.NDArray[np.int64] | None = None
        self.class_labels: list[str] = []
        self._matches: dict[str, list[PredictMatch]] = {}
        self.unfiltered_matches: dict[str, list[PredictMatch]] = {}

        if preprocess:
            self._preprocess()
        if (
            preprocess_preds_conf_threshold is not None
            or preprocess_preds_nms_containment_threshold is not None
            or preprocess_preds_nms_iou_threshold is not None
        ):
            self._preprocess_preds()

    def _validate_df(self, preds_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
        missing_gt = _REQUIRED_COLS_GT - set(gt_df.columns)
        if missing_gt:
            raise ValueError(f"Ground-truth DataFrame is missing columns: {sorted(missing_gt)}")
        missing_preds = _REQUIRED_COLS_PREDS - set(preds_df.columns)
        if missing_preds:
            raise ValueError(f"Predictions DataFrame is missing columns: {sorted(missing_preds)}")

        na_conf = int(preds_df["confidence"].isna().sum())
        if na_conf:
            raise ValueError(
                f"Predictions 'confidence' column contains {na_conf} NA value(s); "
                "every prediction must have a numeric confidence."
            )

        # Predictions must only use classes present in the ground-truth vocabulary.
        gt_classes = set(self.classes)
        pred_classes = set(preds_df["instance_label"].dropna().unique())
        unknown = pred_classes - gt_classes
        if unknown:
            raise ValueError(
                f"Prediction labels not present in ground truth: {sorted(unknown)}. "
                f"Known ground-truth classes: {sorted(gt_classes)}."
            )

    def _preprocess(self) -> None:
        """Remove duplicate GT boxes (based on near-identical IoU)."""
        initial_len = len(self.split_df)
        dups: list[int] = []
        for file_name in self.split_df["image_name"].unique():
            file_df = self.split_df[self.split_df["image_name"] == file_name]
            bboxes = np.array(
                file_df.apply(
                    lambda x: np.array(x[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]]),
                    axis=1,
                ).to_list()
            )
            iou_matrix = compute_iou_matrix(bboxes, bboxes)
            dup_positions, _ = find_duplicates_bboxes(iou_matrix)
            dups.extend(file_df.iloc[dup_positions].index.tolist())

        self.split_df = self.split_df.drop(dups)
        logger.info(f"Preprocessing removed {initial_len - len(self.split_df)} duplicate GT rows.")

    def _preprocess_preds(self) -> None:
        """Apply confidence filtering and/or custom NMS to self.preds_df."""
        if self._preds_conf_threshold is not None:
            n_before = len(self.preds_df)
            self.preds_df = filter_by_confidence(self.preds_df, self._preds_conf_threshold)
            logger.info(
                f"Predictions confidence filtering removed "
                f"{n_before - len(self.preds_df)} rows "
                f"(threshold={self._preds_conf_threshold})."
            )

        cont = self._preds_nms_containment_threshold
        iou = self._preds_nms_iou_threshold
        if cont is not None or iou is not None:
            n_before = len(self.preds_df)
            self.preds_df = apply_nms(
                self.preds_df,
                same_class_containment_threshold=cont if cont is not None else 1.01,
                cross_class_iou_threshold=iou if iou is not None else 1.01,
            )
            logger.info(
                f"Predictions NMS removed {n_before - len(self.preds_df)} rows "
                f"(containment={cont}, iou={iou})."
            )

    def predict_to_dataframe(
        self,
        weights: str,
        *,
        split: str | list[str] | None = None,
        conf: float = 0.001,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | None = None,
        image_name: ImageNameMode = "name",
        **model_kwargs: Any,
    ) -> pd.DataFrame:
        """Generate predictions from a YOLO model over the ground-truth images.

        The image source is ``split_df['image_path']`` (full path to each image),
        so this needs no ``data.yaml``. ``image_name`` is the last part of that
        path, matching the ground-truth ``image_name``. The result is stored as
        ``self.preds_df`` / ``self._raw_preds_df`` (and re-preprocessed if
        confidence/NMS thresholds were configured), so the pipeline can continue
        straight into ``evaluation()``.

        Args:
            weights: Path to Ultralytics model weights (``.pt``).
            split: Restrict inference to one split's images (``"test"``) or several
                (``["test", "val"]``); ``None`` runs over every image in
                ``split_df``. Selecting splits requires a ``"split"`` column.
            conf: Inference confidence threshold (default ``0.001`` keeps the full
                P-R curve available downstream).
            iou: Inference NMS IoU threshold.
            imgsz: Inference image size.
            device: Torch device (e.g. ``"0"``, ``"cpu"``); ``None`` auto-selects.
            image_name: ``image_name`` format — ``"name"`` (filename with
                extension, default), ``"stem"`` or ``"path"``.
            **model_kwargs: Extra keyword arguments forwarded verbatim to
                Ultralytics' ``model.predict`` (e.g. ``half``, ``augment``,
                ``agnostic_nms``, ``max_det``, ``classes``, ``retina_masks``).

        Returns:
            The generated predictions DataFrame (also available as ``self.preds_df``).

        Raises:
            ValueError: If ``split_df`` lacks an ``image_path`` column, lacks a
                ``"split"`` column when ``split`` is given, or has no image paths
                for the requested split(s).
            ImportError: If the optional ``ultralytics`` dependency is missing.
        """
        if split is None:
            gt = self.split_df
        else:
            if "split" not in self.split_df.columns:
                raise ValueError(
                    f"predict_to_dataframe(split={split!r}) requires a 'split' column "
                    "in split_df, but none was found."
                )
            splits = [split] if isinstance(split, str) else list(split)
            gt = self.split_df[self.split_df["split"].isin(splits)]
        if "image_path" not in gt.columns:
            raise ValueError(
                "predict_to_dataframe requires an 'image_path' column in split_df "
                "(full path to each image); none was found."
            )
        image_paths = gt["image_path"].dropna().unique().tolist()
        if not image_paths:
            raise ValueError(f"No 'image_path' values found in split_df (split={split!r}).")

        preds = predict_on_images(
            weights,
            image_paths,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            image_name=image_name,
            **model_kwargs,
        )
        self.preds_df = preds.reset_index(drop=True)
        self._raw_preds_df = self.preds_df.copy()
        self._has_predictions = True
        if (
            self._preds_conf_threshold is not None
            or self._preds_nms_containment_threshold is not None
            or self._preds_nms_iou_threshold is not None
        ):
            self._preprocess_preds()
        return self.preds_df

    @property
    def best_confidences(self) -> dict[str, float]:
        return self._best_confidences

    def __call__(
        self,
        split: str = "all",
        *,
        find_best_confs: bool = True,
        calibration_split: str | None = None,
    ) -> None:
        """Run the evaluation pipeline.

        Args:
            split: Which GT split to evaluate ("all", "train", "val", "test").
            find_best_confs: When True and calibration_split is None, find the
                per-class confidence threshold that maximises F1 on the evaluation
                data itself (in-sample optimisation).  Ignored when
                calibration_split is provided.
            calibration_split: If given (e.g. "val"), optimal thresholds are
                found on that split and then applied to the evaluation split.
                This is the recommended workflow when a held-out validation set
                is available: calibrate on val, report metrics on test.
        """
        self._call(split, find_best_confs=find_best_confs, calibration_split=calibration_split)

    def _define_gt(self, split: str = "all") -> None:
        self.gt_df = (
            deepcopy(self.split_df) if split == "all" else self.split_df.query("split == @split")
        )

    def _splits_to_predict(self, split: str, calibration_split: str | None) -> list[str] | None:
        """Splits whose images must be predicted before the evaluation runs.

        Returns the evaluation split plus the calibration split (when one is
        given), so the auto-predict path covers exactly the splits that will be
        used downstream. Returns ``None`` (every image) when evaluating ``"all"``,
        since that already spans every split.
        """
        if split == "all":
            return None
        splits = [split]
        if calibration_split is not None and calibration_split not in splits:
            splits.append(calibration_split)
        return splits

    def _ensure_predictions(self, splits: list[str] | None = None) -> None:
        """Generate predictions from ``weights_path`` when none were provided.

        Args:
            splits: Restrict auto-prediction to these splits (the evaluation split
                plus any calibration split). ``None`` predicts over every image.

        Raises:
            ValueError: If no predictions exist and no ``weights_path`` was given.
        """
        if self._has_predictions:
            return
        if self._weights_path is None:
            raise ValueError(
                "Evaluation has no predictions to score: preds_df was None and no "
                "weights_path was provided. Pass preds_df, set weights_path=..., or "
                "call predict_to_dataframe() before running the evaluation."
            )
        logger.info(f"Generating predictions from weights '{self._weights_path}'...")
        # Predict only the splits that will be evaluated/calibrated on.
        self.predict_to_dataframe(self._weights_path, split=splits, **self._predict_kwargs)

    def _call(
        self,
        split: str = "all",
        *,
        find_best_confs: bool = True,
        calibration_split: str | None = None,
    ) -> None:
        if self._backend is not None:
            self._run_backend(split, calibration_split=calibration_split)
            return

        self._ensure_predictions(self._splits_to_predict(split, calibration_split))
        self._define_gt(split)
        assert self.gt_df is not None

        self._validate_df(self.preds_df, self.gt_df)

        split_image_names = self.gt_df["image_name"].unique().tolist()

        logger.info("Matching boxes...")
        self._matches = match_boxes(
            self.gt_df,
            self.preds_df,
            self.iou_threshold,
            strategy=self.matching_strategy,
            split_image_names=split_image_names,
        )
        logger.info("Matching complete.")

        if calibration_split is not None:
            self._best_confidences = self._calibrate(calibration_split)
        elif find_best_confs:
            logger.info("Finding best confidence thresholds (in-sample)...")
            self._best_confidences = self._find_confidences(self._matches)
            logger.info("Best thresholds found.")

        logger.info("Filtering by best confidence thresholds...")
        self.unfiltered_matches = self._matches
        self._matches = slice_by_conf(self._matches, self.classes, self._best_confidences)
        logger.info("Filtering complete.")

        logger.info("Computing metrics and confusion matrix...")
        self.metrics = _compute_metrics_from_matches(
            self._matches, self.classes, self._best_confidences
        )
        compute_map(
            self.gt_df,
            self._raw_preds_df,
            self.metrics,
            split_image_names,
            method=self._ap_method,
            strategy=self.matching_strategy,
        )
        self._compute_cohen_kappa()
        self.cm, self.class_labels = get_confusion_matrix(self._matches, self.classes)
        logger.info("Metrics and confusion matrix computed.")

    def compute_metrics_ultralytics(self, split: str = "all") -> dict[str, DetectionMetrics]:
        """Score ``split`` with the Ultralytics (YOLO-comparable) backend.

        Generates predictions first if needed (``weights_path`` flow) and runs
        over the raw, unpreprocessed predictions. Returns per-class
        :class:`DetectionMetrics`; requires the ``ultralytics`` extra.
        """
        return self._compute_external("ultralytics", split)

    def compute_metrics_torchmetrics(self, split: str = "all") -> dict[str, DetectionMetrics]:
        """Score ``split`` with the torchmetrics (COCO mAP) backend.

        Generates predictions first if needed (``weights_path`` flow) and runs
        over the raw, unpreprocessed predictions. Returns per-class
        :class:`DetectionMetrics`; requires the ``torchmetrics`` extra.
        """
        return self._compute_external("torchmetrics", split)

    def _compute_external(self, backend: Backend, split: str) -> dict[str, DetectionMetrics]:
        """Run an external backend over ``split`` and return its per-class metrics."""
        self._ensure_predictions(self._splits_to_predict(split, None))
        self._define_gt(split)
        assert self.gt_df is not None
        # Backends score the raw predictions (YOLO val style); no conf/NMS preprocessing.
        self._validate_df(self._raw_preds_df, self.gt_df)
        split_image_names = self.gt_df["image_name"].unique().tolist()
        logger.info(f"Computing metrics with the '{backend}' backend on split '{split}'...")
        return compute_detection_metrics(
            self.gt_df,
            self._raw_preds_df,
            backend=backend,
            classes=self.classes,
            split_image_names=split_image_names,
        )

    def _run_backend(self, split: str, *, calibration_split: str | None = None) -> None:
        """Populate ``detection_metrics``/``metrics`` from the selected backend.

        The raw backend output is kept on ``detection_metrics``; ``metrics`` holds
        the same numbers adapted to native :class:`Metrics` so the dashboards and
        CI plots work unchanged. The ``"ultralytics"`` backend also fills the
        confusion matrix (via Ultralytics' own ``ConfusionMatrix`` logic);
        ``"torchmetrics"`` has no confusion matrix, so it is cleared.

        When ``calibration_split`` is given, the ``"ultralytics"`` backend reads
        P/R/F1 at the confidence calibrated on that split (AP stays over the full
        curve). ``"torchmetrics"`` does not support calibration yet, so the split
        is ignored with a warning and the backend self-selects.
        """
        assert self._backend is not None
        if calibration_split is not None and self._backend != "ultralytics":
            logger.warning(
                f"calibration_split={calibration_split!r} is not supported for "
                f"backend={self._backend!r} yet; ignoring it (the backend self-selects "
                "its operating point)."
            )
            calibration_split = None

        if calibration_split is None:
            self.detection_metrics = self._compute_external(self._backend, split)
        else:
            self.detection_metrics = self._calibrate_backend(split, calibration_split)
        self.metrics = self._adapt_detection_metrics(self.detection_metrics)
        if self._backend == "ultralytics":
            assert self.gt_df is not None
            split_image_names = self.gt_df["image_name"].unique().tolist()
            self.cm, self.class_labels = compute_ultralytics_confusion_matrix(
                self.gt_df,
                self._raw_preds_df,
                classes=self.classes,
                split_image_names=split_image_names,
            )
        else:
            self.cm = None
            self.class_labels = []

    def _adapt_detection_metrics(
        self, detection_metrics: dict[str, DetectionMetrics]
    ) -> dict[str, Metrics]:
        """Map external ``DetectionMetrics`` onto native ``Metrics`` for the dashboards.

        The backends report only precision/recall/f1 and AP at a self-selected
        operating point — no box-level TP/FP/FN. We reconstruct float counts from
        the per-class ground-truth size ``N`` (= TP + FN, known from ``gt_df``) so
        the reproduced precision/recall/f1 equal the backend's exactly::

            TP = recall * N        FN = N - TP        FP = TP * (1 - p) / p   (p > 0)

        Wilson CIs then follow from these counts — the recall CI is grounded in the
        true ``N``; the precision CI is approximate because FP is reconstructed, not
        counted. ``cohen_kappa`` is set to ``-1`` (not provided by external
        backends) and the confidence threshold to ``0.0`` (the operating point is
        internal to the backend). Classes with no GT in the split get NaN AP,
        matching the native convention.
        """
        assert self.gt_df is not None
        gt_counts = self.gt_df["instance_label"].value_counts().to_dict()
        result: dict[str, Metrics] = {}
        for c in self.classes:
            n_gt = int(gt_counts.get(c, 0))
            dm = detection_metrics.get(c)
            if dm is None or n_gt == 0:
                result[c] = Metrics(
                    ap50=float("nan"),
                    ap75=float("nan"),
                    ap50_95=float("nan"),
                    cohen_kappa=-1,
                )
                continue
            tp = dm.recall * n_gt
            fn = n_gt - tp
            fp = tp * (1.0 - dm.precision) / dm.precision if dm.precision > 0 else 0.0
            result[c] = Metrics(
                tp=tp,
                fp=fp,
                fn=fn,
                ap50=dm.ap50,
                ap75=dm.ap75,
                ap50_95=dm.ap50_95,
                cohen_kappa=-1,
            )
        return result

    def _calibration_gt(self, calibration_split: str) -> pd.DataFrame:
        """Return the validated ground truth for *calibration_split*.

        Shared by the native and backend calibration paths. ``self.gt_df`` (the
        evaluation split) must already be set, so leakage can be detected.

        Raises:
            ValueError: If split_df has no "split" column, the calibration split
                has no rows, or it shares an ``image_name`` with the evaluation
                split (which would leak calibration data into the evaluation).
        """
        if "split" not in self.split_df.columns:
            raise ValueError(
                f"calibration_split={calibration_split!r} requires split_df to have "
                "a 'split' column, but none was found."
            )
        cal_gt = self.split_df[self.split_df["split"] == calibration_split]
        if cal_gt.empty:
            available = self.split_df["split"].unique().tolist()
            raise ValueError(
                f"No ground-truth rows found for calibration split {calibration_split!r}. "
                f"Available splits: {available}"
            )

        assert self.gt_df is not None
        overlap = set(cal_gt["image_name"]) & set(self.gt_df["image_name"])
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"Calibration split {calibration_split!r} shares "
                f"{len(overlap)} image_name(s) with the evaluation split "
                f"(e.g. {sample}). Predictions are matched to ground truth via "
                "image_name, so overlapping images would leak calibration data "
                "into the evaluation. Fix the 'split' labels in split_df so each "
                "image_name belongs to exactly one split."
            )
        return cal_gt

    def _calibrate_backend(self, split: str, calibration_split: str) -> dict[str, DetectionMetrics]:
        """Ultralytics backend metrics with the operating point calibrated on val.

        Finds the F1-optimal confidence on ``calibration_split`` (per the
        configured ``confidence_optimization`` mode), then reads the eval split's
        P/R/F1 at that confidence while AP stays over the full curve. Also records
        the chosen threshold(s) on ``best_confidences``.
        """
        self._ensure_predictions(self._splits_to_predict(split, calibration_split))
        self._define_gt(split)
        assert self.gt_df is not None
        self._validate_df(self._raw_preds_df, self.gt_df)
        cal_gt = self._calibration_gt(calibration_split)

        cal_image_names = cal_gt["image_name"].unique().tolist()
        logger.info(
            f"Calibrating '{self._backend}' confidence on '{calibration_split}' "
            f"({len(cal_gt)} GT rows, mode={self._confidence_optimization})..."
        )
        conf = find_ultralytics_confidence(
            cal_gt,
            self._raw_preds_df,
            classes=self.classes,
            split_image_names=cal_image_names,
            mode=self._confidence_optimization,
        )
        if isinstance(conf, dict):
            self._best_confidences = {c: conf.get(c, 0.0) for c in self.classes}
        else:
            self._best_confidences = {c: conf for c in self.classes}

        split_image_names = self.gt_df["image_name"].unique().tolist()
        return compute_ultralytics_metrics(
            self.gt_df,
            self._raw_preds_df,
            classes=self.classes,
            split_image_names=split_image_names,
            conf_threshold=conf,
        )

    def _calibrate(self, calibration_split: str) -> dict[str, float]:
        """Find per-class confidence thresholds on *calibration_split*.

        Args:
            calibration_split: Name of the split column value to calibrate on
                (e.g. "val"). split_df must have a "split" column.

        Returns:
            Dict mapping class name → best confidence threshold.

        Raises:
            ValueError: If split_df has no "split" column, or if
                calibration_split has no matching rows.
        """
        cal_gt = self._calibration_gt(calibration_split)
        cal_image_names = cal_gt["image_name"].unique().tolist()
        logger.info(
            f"Calibrating confidence thresholds on '{calibration_split}' split "
            f"({len(cal_gt)} GT rows)..."
        )
        cal_matches = match_boxes(
            cal_gt,
            self.preds_df,
            self.iou_threshold,
            strategy=self.matching_strategy,
            split_image_names=cal_image_names,
        )
        thresholds = self._find_confidences(cal_matches)
        logger.info("Threshold calibration complete.")
        return thresholds

    def _find_confidences(self, matches: dict[str, list[PredictMatch]]) -> dict[str, float]:
        """Choose confidence thresholds per the configured optimisation mode.

        ``"per_class"`` returns an independent threshold per class;
        ``"global"`` returns the same YOLO-style threshold for every class.
        """
        if self._confidence_optimization == "global":
            threshold = find_best_global_confidence(matches, self.classes)
            thresholds = {c: threshold for c in self.classes}
        else:
            thresholds = find_best_confidences(matches, self.classes)
        self._warn_if_thresholds_unoptimized(matches, thresholds)
        return thresholds

    def _warn_if_thresholds_unoptimized(
        self, matches: dict[str, list[PredictMatch]], thresholds: dict[str, float]
    ) -> None:
        """Warn when an optimised threshold keeps every detection.

        A threshold equal to the minimum prediction confidence does not discard
        anything, so confidence optimisation had no effect — typically because the
        predictions match the ground truth so well that the optimal cut is none
        (e.g. identical pred/GT boxes, where the F1-optimal threshold is the floor).
        """

        def min_confidence(records: list[PredictMatch]) -> float | None:
            confs = [m.confidence for m in records if m.type != "FN"]
            return min(confs) if confs else None

        if self._confidence_optimization == "global":
            all_confs = [m.confidence for recs in matches.values() for m in recs if m.type != "FN"]
            if not all_confs:
                return
            global_min = min(all_confs)
            threshold = next(iter(thresholds.values()), 0.0)
            if threshold <= global_min:
                logger.warning(
                    f"Global confidence threshold ({threshold:.6g}) equals the minimum "
                    f"prediction confidence ({global_min:.6g}); it keeps every detection, "
                    "so confidence optimisation had no effect (predictions may match GT "
                    "closely)."
                )
            return

        for c in self.classes:
            mc = min_confidence(matches.get(c, []))
            if mc is None:
                continue
            if thresholds.get(c, 0.0) <= mc:
                logger.warning(
                    f"Confidence threshold for class '{c}' ({thresholds[c]:.6g}) equals the "
                    f"minimum prediction confidence ({mc:.6g}); it keeps every detection, so "
                    "confidence optimisation had no effect for this class."
                )

    def _compute_cohen_kappa(self) -> None:
        # Sentinel -1 for every class (including any absent from this split), so
        # the column is uniform when kappa is skipped.
        if self._skip_cohen_kappa:
            for m in self.metrics.values():
                m.cohen_kappa = -1
            return

        assert self.gt_df is not None
        missing = {"image_width", "image_height"} - set(self.gt_df.columns)
        if missing:
            raise ValueError(
                f"Cohen's kappa needs the optional column(s) {sorted(missing)} in the "
                "ground-truth DataFrame (image pixel dimensions for the masks). Add them "
                "or keep skip_cohen_kappa=True."
            )

        bbox_cols = ["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]
        for c in tqdm(
            self.gt_df["instance_label"].unique(),
            desc="Computing Cohen's Kappa",
            total=self.gt_df["instance_label"].nunique(),
        ):
            kappas: list[float] = []
            class_gt = self.gt_df[self.gt_df["instance_label"] == c]
            preds_gt = self.preds_df[self.preds_df["instance_label"] == c]

            for image_name in class_gt["image_name"].unique():
                gt_boxes = class_gt[class_gt["image_name"] == image_name]
                pred_boxes = preds_gt[preds_gt["image_name"] == image_name]
                # Plain (n, 4) arrays: compute_kappa indexes each box positionally.
                gt_box_list = gt_boxes[bbox_cols].to_numpy(np.float64)
                pred_box_list = pred_boxes[bbox_cols].to_numpy(np.float64)
                # Use this image's own dimensions, not the first image's.
                kappa = compute_kappa(
                    gt_box_list,
                    pred_box_list,
                    (int(gt_boxes.iloc[0]["image_width"]), int(gt_boxes.iloc[0]["image_height"])),
                )
                kappas.append(kappa)

            self.metrics[c].cohen_kappa = float(np.mean(kappas)) if kappas else -1.0
            logger.debug(f"Kappa for {c}: {self.metrics[c].cohen_kappa}")

    def _get_metrics_as_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {k: v.model_dump() for k, v in self.metrics.items()}, orient="index"
        )

    def get_dashboards(
        self,
        *,
        save_to_excel: bool = True,
        path: str = "metrics/",
        save_confusion_matrix: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate analyst and production dashboards.

        Args:
            save_to_excel: Whether to write Excel files. Defaults to True.
            path: Output directory. Defaults to "metrics/".
            save_confusion_matrix: Whether to save the CM to Excel. Defaults to True.

        Returns:
            (devs, dtrk) DataFrames.
        """
        assert self.metrics, "Call evaluation() before get_dashboards()."
        return get_dashboards(
            metrics=self.metrics,
            split_df=self.split_df,
            cm=self.cm,
            class_labels=self.class_labels,
            suffix=self.suffix,
            save_to_excel=save_to_excel,
            path=path,
            save_confusion_matrix=save_confusion_matrix,
        )

    def plot_confidence_intervals(
        self,
        metric: str,
        confidence_level: float = 0.95,
        save_path: str | None = None,
        figsize: tuple[int, int] = (10, 6),
    ) -> tuple[Figure | None, Axes | None]:
        """Plot CI bar chart for a metric.

        Args:
            metric: One of "precision", "recall", "perebrak", "nedobrak".
            confidence_level: CI level passed through for the title. Defaults to 0.95.
            save_path: Where to save the PNG; None shows the plot.
            figsize: Figure size.

        Returns:
            (fig, ax) matplotlib objects.
        """
        return plot_confidence_intervals(
            metrics=self.metrics,
            metric=metric,
            confidence_level=confidence_level,
            save_path=save_path,
            figsize=figsize,
        )

    def get_topk_confusions(self, main_class: str, k: int = 5) -> pd.DataFrame:
        """Return a DataFrame of annotation records for the top-k confused classes.

        Args:
            main_class: Class to audit.
            k: Number of top confused classes to include.

        Returns:
            DataFrame with box records for visual inspection.
        """
        if self.cm is None:
            if self._backend is not None:
                raise ValueError(
                    f"get_topk_confusions needs a confusion matrix, but the "
                    f"'{self._backend}' backend does not produce one."
                )
            logger.info("Confusion matrix not yet computed; running evaluation.")
            self._call()

        assert self.cm is not None
        class_index = self.class_labels.index(main_class)
        confusion_counts = self.cm[class_index, :].flatten() + self.cm[:, class_index].flatten()

        confusion_df = pd.DataFrame(confusion_counts, index=self.class_labels, columns=["count"])
        # Exclude the class itself and the background bucket from the confusions.
        confusion_df = confusion_df.drop(index=[main_class, "background"], errors="ignore")
        top_k = confusion_df.nlargest(k, "count")
        logger.info(f"Top {k} confusions for {main_class!r}: {top_k.index.tolist()}")

        assert self.gt_df is not None
        return get_confusions(
            matches=self._matches,
            class_labels=self.class_labels,
            preds_df=self.preds_df,
            gt_df=self.gt_df,
            main_class=main_class,
            subclasses=top_k.index.tolist(),
        )

    def get_dfs_visualization(
        self, *, find_best_confs: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return GT and preds DataFrames annotated with prediction type.

        Args:
            find_best_confs: Whether to run confidence optimisation if not done yet.

        Returns:
            (gt_df, preds_df) with a "predict_type" column.
        """
        if self.gt_df is None:
            logger.info("Ground-truth DataFrame not available; running evaluation.")
            self._call(find_best_confs=find_best_confs)

        assert self.gt_df is not None

        # Annotate from the match records (before confidence slicing, so every
        # prediction and every GT box is classified). Predictions are keyed by
        # pred_index, GT boxes by gt_index; GT takes TP/FN only (a cross-class FP
        # also references a GT index, but that GT's own status is its TP/FN record).
        matches = self.unfiltered_matches or self._matches
        pred_type: dict[int, str] = {}
        gt_type: dict[int, str] = {}
        for records in matches.values():
            for m in records:
                if m.pred_index != -1:
                    pred_type[m.pred_index] = m.type
                if m.gt_index != -1 and m.type in ("TP", "FN"):
                    gt_type[m.gt_index] = m.type

        gt_df = self.gt_df.copy()
        preds_df = self.preds_df.copy()
        gt_df["predict_type"] = [gt_type.get(i, "FN") for i in gt_df.index]
        preds_df["predict_type"] = [pred_type.get(i, "FP") for i in preds_df.index]
        return gt_df, preds_df
