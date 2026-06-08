from __future__ import annotations

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm

from .ap import compute_map
from .confidence import find_best_confidences, slice_by_conf
from .confusion import get_confusion_matrix, get_confusions
from .dashboard import get_dashboards, plot_confidence_intervals
from .iou import compute_iou_matrix, find_duplicates_bboxes
from .kappa import compute_kappa
from .matching import MatchingStrategy, match_boxes
from .types import Metrics, PredictMatch

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
        preds_df: pd.DataFrame | str,
        split_df: pd.DataFrame | str,
        iou_threshold: float = 0.5,
        preprocess: bool = False,
        skip_cohen_kappa: bool = True,
        matching_strategy: MatchingStrategy = "greedy",
    ) -> None:
        """Initialise the Evaluation object.

        Args:
            preds_df: Predictions DataFrame or path to CSV.
            split_df: Ground-truth split DataFrame or path to CSV.
            iou_threshold: IoU threshold for box matching. Defaults to 0.5.
            preprocess: Remove duplicate GT boxes if True. Defaults to False.
            skip_cohen_kappa: Skip cohen_kappa computation. Defaults to True.
            matching_strategy: "greedy" (default, YOLO-style) or "hungarian".
        """
        self.suffix = "default"

        if isinstance(preds_df, str):
            preds_df = pd.read_csv(preds_df)
        if isinstance(split_df, str):
            split_df = pd.read_csv(split_df)

        self.preds_df: pd.DataFrame = preds_df.reset_index(drop=True)
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

        self.metrics: dict[str, Metrics] = {}
        self.cm: npt.NDArray[np.int64] | None = None
        self.class_labels: list[str] = []
        self._matches: dict[str, list[PredictMatch]] = {}
        self.unfiltered_matches: dict[str, list[PredictMatch]] = {}

        if preprocess:
            self._preprocess()

    def _validate_df(self, preds_df: pd.DataFrame, gt_df: pd.DataFrame) -> None:
        missing_gt = _REQUIRED_COLS_GT - set(gt_df.columns)
        if missing_gt:
            raise ValueError(f"Ground-truth DataFrame is missing columns: {sorted(missing_gt)}")
        missing_preds = _REQUIRED_COLS_PREDS - set(preds_df.columns)
        if missing_preds:
            raise ValueError(f"Predictions DataFrame is missing columns: {sorted(missing_preds)}")

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

    @property
    def best_confidences(self) -> dict[str, float]:
        return self._best_confidences

    def __call__(
        self,
        split: str = "all",
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
        self._call(split, find_best_confs, calibration_split)

    def _define_gt(self, split: str = "all") -> None:
        self.gt_df = (
            deepcopy(self.split_df) if split == "all" else self.split_df.query("split == @split")
        )

    def _call(
        self,
        split: str = "all",
        find_best_confs: bool = True,
        calibration_split: str | None = None,
    ) -> None:
        self._define_gt(split)
        assert self.gt_df is not None

        self._validate_df(self.preds_df, self.gt_df)

        logger.info("Matching boxes...")
        self._matches = match_boxes(
            self.gt_df, self.preds_df, self.iou_threshold, strategy=self.matching_strategy
        )
        logger.info("Matching complete.")

        if calibration_split is not None:
            self._best_confidences = self._calibrate(calibration_split)
        elif find_best_confs:
            logger.info("Finding best confidence thresholds (in-sample)...")
            self._best_confidences = find_best_confidences(self._matches, self.classes)
            logger.info("Best thresholds found.")

        logger.info("Filtering by best confidence thresholds...")
        self.unfiltered_matches = self._matches
        self._matches = slice_by_conf(self._matches, self.classes, self._best_confidences)
        logger.info("Filtering complete.")

        logger.info("Computing metrics and confusion matrix...")
        self.metrics = _compute_metrics_from_matches(
            self._matches, self.classes, self._best_confidences
        )
        compute_map(self.gt_df, self.preds_df, self.metrics)
        self._compute_cohen_kappa()
        self.cm, self.class_labels = get_confusion_matrix(self._matches, self.classes)
        logger.info("Metrics and confusion matrix computed.")

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

        logger.info(
            f"Calibrating confidence thresholds on '{calibration_split}' split "
            f"({len(cal_gt)} GT rows)..."
        )
        cal_matches = match_boxes(
            cal_gt, self.preds_df, self.iou_threshold, strategy=self.matching_strategy
        )
        thresholds = find_best_confidences(cal_matches, self.classes)
        logger.info("Threshold calibration complete.")
        return thresholds

    def _compute_cohen_kappa(self) -> None:
        assert self.gt_df is not None
        for c in tqdm(
            self.gt_df["instance_label"].unique(),
            desc="Computing Cohen's Kappa",
            total=self.gt_df["instance_label"].nunique(),
        ):
            if self._skip_cohen_kappa:
                self.metrics[c].cohen_kappa = -1
                continue

            kappas: list[float] = []
            class_gt = self.gt_df[self.gt_df["instance_label"] == c].copy()
            preds_gt = self.preds_df[self.preds_df["instance_label"] == c].copy()

            for image_name in class_gt["image_name"].unique():
                gt_boxes = class_gt[class_gt["image_name"] == image_name].copy()
                gt_box_list = [
                    row[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]]
                    for _, row in gt_boxes.iterrows()
                ]
                pred_boxes = preds_gt[preds_gt["image_name"] == image_name].copy()
                pred_box_list = [
                    row[["bbox_x_tl", "bbox_y_tl", "bbox_x_br", "bbox_y_br"]]
                    for _, row in pred_boxes.iterrows()
                ]
                kappa = compute_kappa(
                    gt_box_list,
                    pred_box_list,
                    (int(class_gt.iloc[0]["image_width"]), int(class_gt.iloc[0]["image_height"])),
                )
                kappas.append(kappa)

            self.metrics[c].cohen_kappa = float(np.mean(kappas))
            logger.debug(f"Kappa for {c}: {self.metrics[c].cohen_kappa}")

    def _get_metrics_as_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {k: v.model_dump() for k, v in self.metrics.items()}, orient="index"
        )

    def get_dashboards(
        self,
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
        assert self.cm is not None, "Call evaluation() before get_dashboards()."
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
            logger.info("Confusion matrix not yet computed; running evaluation.")
            self._call()

        assert self.cm is not None
        class_index = self.class_labels.index(main_class)
        confusion_counts = self.cm[class_index, :].flatten() + self.cm[:, class_index].flatten()

        confusion_df = pd.DataFrame(confusion_counts, index=self.class_labels, columns=["count"])
        confusion_df = confusion_df.drop(main_class)
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
        self, find_best_confs: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return GT and preds DataFrames annotated with prediction type.

        Args:
            find_best_confs: Whether to run confidence optimisation if not done yet.

        Returns:
            (gt_df, preds_df) with a "predict_type" column.
        """
        if self.gt_df is None:
            self._call(find_best_confs=find_best_confs)
            logger.info("Ground-truth DataFrame not available; running evaluation.")

        assert self.gt_df is not None
        self.gt_df["predict_type"] = "TP"
        self.preds_df["predict_type"] = "TP"

        for match_list in self._matches.values():
            for match in match_list:
                if match.type == "FN":
                    self.gt_df.loc[match.gt_index, "predict_type"] = match.type
                else:
                    self.preds_df.loc[match.pred_index, "predict_type"] = match.type

        return self.gt_df, self.preds_df
