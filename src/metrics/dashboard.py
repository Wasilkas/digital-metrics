from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .types import Metrics


def _metrics_as_df(metrics: dict[str, Metrics]) -> pd.DataFrame:
    metrics_dict = {k: v.model_dump() for k, v in metrics.items()}
    return pd.DataFrame.from_dict(metrics_dict, orient="index")


def get_dashboards(
    metrics: dict[str, Metrics],
    split_df: pd.DataFrame,
    cm: npt.NDArray[np.int64],
    class_labels: list[str],
    suffix: str = "default",
    save_to_excel: bool = True,
    path: str = "metrics/",
    save_confusion_matrix: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate analyst and production dashboards; optionally save to Excel.

    Args:
        metrics: Per-class Metrics objects.
        split_df: Ground-truth split DataFrame (used for example counts).
        cm: Confusion matrix array.
        class_labels: Class label list including background.
        suffix: Filename suffix for saved files.
        save_to_excel: Whether to write Excel files.
        path: Output directory.
        save_confusion_matrix: Whether to save the confusion matrix to Excel.

    Returns:
        (devs, dtrk) — full analyst dashboard and production dashboard.
    """
    for metric_name in ("recall", "precision", "perebrak", "nedobrak"):
        plot_confidence_intervals(
            metrics=metrics,
            metric=metric_name,
            confidence_level=0.95,
            save_path=os.path.join(path, f"{metric_name}_confidence_intervals.png"),
            figsize=(12, 8),
        )

    metrics_df = _metrics_as_df(metrics)
    metrics_df["confidence"] = metrics_df["confidence"].apply(lambda x: round(x, 2))

    if "split" in split_df.columns:
        counts = split_df.groupby(["instance_label", "split"]).size().unstack(fill_value=0)
        for s in ("train", "test", "val"):
            if s not in counts.columns:
                counts[s] = 0
        counts = counts.rename(
            columns={
                "train": "Количество примеров train",
                "test": "Количество примеров test",
                "val": "Количество примеров val",
            }
        )
    else:
        counts = (
            split_df.groupby(["instance_label"])
            .size()
            .reset_index(name="Количество примеров")
            .set_index("instance_label")
        )

    devs = metrics_df.merge(counts, how="left", left_index=True, right_index=True)
    dtrk = (
        metrics_df[
            [
                "nedobrak",
                "nedobrak_ci_lower",
                "nedobrak_ci_upper",
                "perebrak",
                "perebrak_ci_lower",
                "perebrak_ci_upper",
                "confidence",
            ]
        ]
        .merge(counts, how="left", left_index=True, right_index=True)
        .rename(
            columns={
                "confidence": "Порог",
                "perebrak": "Перебраковка",
                "nedobrak": "Недобраковка",
            }
        )
    )

    dtrk["Недобраковка"] = dtrk["Недобраковка"] * 100
    dtrk["Перебраковка"] = dtrk["Перебраковка"] * 100
    dtrk["nedobrak_ci_lower"] = dtrk["nedobrak_ci_lower"] * 100
    dtrk["nedobrak_ci_upper"] = dtrk["nedobrak_ci_upper"] * 100
    dtrk["perebrak_ci_lower"] = dtrk["perebrak_ci_lower"] * 100
    dtrk["perebrak_ci_upper"] = dtrk["perebrak_ci_upper"] * 100

    devs = devs.loc[:, ~devs.columns.str.contains("^Unnamed")].sort_index()
    dtrk = dtrk.loc[:, ~dtrk.columns.str.contains("^Unnamed")].sort_index()

    logger.info("Saving results...")
    os.makedirs(path, exist_ok=True)

    if save_confusion_matrix:
        cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
        cm_df.to_excel(os.path.join(path, f"matrix_{suffix}.xlsx"))

    if save_to_excel:
        devs.to_excel(os.path.join(path, f"full_dashboard_{suffix}.xlsx"))
        dtrk.to_excel(os.path.join(path, f"метрики_дтрк_{suffix}.xlsx"))

    logger.info("Save complete.")
    return devs, dtrk


def plot_confidence_intervals(
    metrics: dict[str, Metrics],
    metric: str,
    confidence_level: float = 0.95,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> tuple[Figure | None, Axes | None]:
    """Plot bar chart with Wilson CI error bars for a metric across classes.

    Args:
        metrics: Per-class Metrics objects.
        metric: One of "precision", "recall", "perebrak", "nedobrak".
        confidence_level: Desired CI level (unused here; CIs come from Metrics).
        save_path: Path to save the PNG; if None, the plot is displayed.
        figsize: Figure size in inches.

    Returns:
        (fig, ax) matplotlib objects, or (None, None) if metrics is empty.
    """
    if not metrics:
        logger.error("Metrics have not been computed yet. Run evaluation first.")
        return None, None

    class_names = list(metrics.keys())
    means: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []

    for class_name in class_names:
        m = metrics[class_name]
        if metric == "precision":
            val, lower, upper = m.precision, m.precision_ci_lower, m.precision_ci_upper
        elif metric == "recall":
            val, lower, upper = m.recall, m.recall_ci_lower, m.recall_ci_upper
        elif metric == "perebrak":
            val, lower, upper = m.perebrak, m.perebrak_ci_lower, m.perebrak_ci_upper
        elif metric == "nedobrak":
            val, lower, upper = m.nedobrak, m.nedobrak_ci_lower, m.nedobrak_ci_upper
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

        means.append(val)
        lowers.append(max(0.0, val - lower))
        uppers.append(0.0 if np.isnan(upper) or (upper - val) <= 0 else upper - val)

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(class_names))

    ax.bar(x_pos, means, yerr=[lowers, uppers], align="center", ecolor="black", capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel(f"{metric.upper()} Score")
    ax.set_xlabel("Class")
    title = f"{int(confidence_level * 100)}% Confidence Intervals for {metric.upper()} by Class"
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info(f"Confidence interval plot saved to {save_path}")
    else:
        plt.show()

    return fig, ax
