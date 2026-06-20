from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .backends import Backend, compute_detection_metrics
from .calibration import ConfidenceCalibrator
from .config import InferenceConfig, PreprocessConfig, ScoringConfig
from .engines import BackendEngine, NativeEngine, ScoringEngine, ScoringInputs
from .inference import ImageNameMode, predict_on_images
from .matching import MatchingStrategy, compute_iou_matrix, find_duplicates_bboxes
from .preprocess import PredictionPreprocessor
from .reporting import get_dashboards, plot_confidence_intervals
from .scoring import APMethod, ConfidenceOptimization, get_confusions
from .types import DetectionMetrics, Metrics, PredictMatch
from .validation import REQUIRED_COLS_PREDS, validate_dataframes


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
        scoring: ScoringConfig | None = None,
        preprocessing: PreprocessConfig | None = None,
        inference: InferenceConfig | None = None,
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
            scoring: Optional :class:`~metrics.config.ScoringConfig` grouping
                ``iou_threshold`` / ``matching_strategy`` / ``ap_method`` /
                ``confidence_optimization`` / ``skip_cohen_kappa``. When given it
                takes precedence over those flat kwargs.
            preprocessing: Optional :class:`~metrics.config.PreprocessConfig`
                grouping the GT dedup flag (``dedup_gt`` ← ``preprocess``) and the
                predictions conf/NMS thresholds. When given it takes precedence
                over those flat kwargs.
            inference: Optional :class:`~metrics.config.InferenceConfig` grouping
                ``weights_path`` / ``predict_kwargs``. When given it takes
                precedence over those flat kwargs.
        """
        self.suffix = "default"

        if isinstance(preds_df, str):
            preds_df = pd.read_csv(preds_df)
        if isinstance(split_df, str):
            split_df = pd.read_csv(split_df)

        # A grouped config, when provided, supplies its whole group and takes
        # precedence over the corresponding flat kwargs (which set the defaults).
        scoring = scoring or ScoringConfig(
            iou_threshold=iou_threshold,
            matching_strategy=matching_strategy,
            ap_method=ap_method,
            confidence_optimization=confidence_optimization,
            skip_cohen_kappa=skip_cohen_kappa,
        )
        preprocessing = preprocessing or PreprocessConfig(
            dedup_gt=preprocess,
            conf_threshold=preprocess_preds_conf_threshold,
            nms_containment_threshold=preprocess_preds_nms_containment_threshold,
            nms_iou_threshold=preprocess_preds_nms_iou_threshold,
        )
        inference = inference or InferenceConfig(
            weights_path=weights_path,
            predict_kwargs=predict_kwargs,
        )

        self._weights_path = inference.weights_path
        self._has_predictions = preds_df is not None
        if not self._has_predictions and inference.weights_path is not None:
            logger.info(
                f"No predictions provided; they will be generated from weights "
                f"'{inference.weights_path}' when the evaluation runs."
            )
        if self._has_predictions and inference.weights_path is not None:
            logger.warning(
                "Both preds_df and weights_path were provided; using preds_df and "
                "ignoring weights_path."
            )
        if preds_df is None:
            # Placeholder until predictions are generated (predict_to_dataframe).
            preds_df = pd.DataFrame(columns=sorted(REQUIRED_COLS_PREDS))

        self.preds_df: pd.DataFrame = preds_df.reset_index(drop=True)
        self._raw_preds_df: pd.DataFrame = self.preds_df.copy()
        self.split_df: pd.DataFrame = split_df.reset_index(drop=True)
        self.gt_df: pd.DataFrame | None = None

        self.iou_threshold = scoring.iou_threshold
        # Defer KeyError: validate_dataframes raises ValueError with a clear message if missing.
        self.classes: list[str] = (
            self.split_df["instance_label"].unique().tolist()
            if "instance_label" in self.split_df.columns
            else []
        )
        self._best_confidences: dict[str, float] = {c: 0.0 for c in self.classes}
        self._skip_cohen_kappa = scoring.skip_cohen_kappa
        self.matching_strategy: MatchingStrategy = scoring.matching_strategy
        self._preprocessor = PredictionPreprocessor(
            conf_threshold=preprocessing.conf_threshold,
            nms_containment_threshold=preprocessing.nms_containment_threshold,
            nms_iou_threshold=preprocessing.nms_iou_threshold,
        )
        self._ap_method: APMethod = scoring.ap_method
        self._confidence_optimization: ConfidenceOptimization = scoring.confidence_optimization
        self._backend: Backend | None = backend
        self._predict_kwargs: dict[str, Any] = inference.predict_kwargs or {}
        self._calibrator = ConfidenceCalibrator(
            classes=self.classes,
            iou_threshold=self.iou_threshold,
            matching_strategy=self.matching_strategy,
            confidence_optimization=self._confidence_optimization,
        )
        self._engine: ScoringEngine = (
            NativeEngine(
                classes=self.classes,
                iou_threshold=self.iou_threshold,
                matching_strategy=self.matching_strategy,
                ap_method=self._ap_method,
                skip_cohen_kappa=self._skip_cohen_kappa,
                calibrator=self._calibrator,
            )
            if backend is None
            else BackendEngine(
                backend=backend,
                classes=self.classes,
                confidence_optimization=self._confidence_optimization,
                calibrator=self._calibrator,
            )
        )

        self.metrics: dict[str, Metrics] = {}
        self.detection_metrics: dict[str, DetectionMetrics] = {}
        self.cm: npt.NDArray[np.int64] | None = None
        self.class_labels: list[str] = []
        self._matches: dict[str, list[PredictMatch]] = {}
        self.unfiltered_matches: dict[str, list[PredictMatch]] = {}

        if preprocessing.dedup_gt:
            self._preprocess()
        if self._preprocessor.enabled:
            self.preds_df = self._preprocessor.process(self.preds_df)

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
        if self._preprocessor.enabled:
            self.preds_df = self._preprocessor.process(self.preds_df)
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
        # The engine may decline a calibration split (e.g. torchmetrics); resolve
        # it first so auto-prediction covers exactly the splits that get used.
        calibration_split = self._engine.resolve_calibration_split(calibration_split)
        self._ensure_predictions(self._splits_to_predict(split, calibration_split))
        self._define_gt(split)
        assert self.gt_df is not None

        result = self._engine.run(
            ScoringInputs(
                gt_df=self.gt_df,
                preds_df=self.preds_df,
                raw_preds_df=self._raw_preds_df,
                split_df=self.split_df,
                split=split,
                find_best_confs=find_best_confs,
                calibration_split=calibration_split,
            )
        )
        self.metrics = result.metrics
        self._best_confidences = result.best_confidences
        self.cm = result.cm
        self.class_labels = result.class_labels
        self.detection_metrics = result.detection_metrics
        self._matches = result.matches
        self.unfiltered_matches = result.unfiltered_matches

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
        validate_dataframes(self._raw_preds_df, self.gt_df, self.classes)
        split_image_names = self.gt_df["image_name"].unique().tolist()
        logger.info(f"Computing metrics with the '{backend}' backend on split '{split}'...")
        return compute_detection_metrics(
            self.gt_df,
            self._raw_preds_df,
            backend=backend,
            classes=self.classes,
            split_image_names=split_image_names,
        )

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
