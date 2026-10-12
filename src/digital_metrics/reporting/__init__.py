"""Reporting: Excel dashboards and confidence-interval plots."""

from .dashboard import (
    RenderedEvaluation,
    RenderedPlot,
    get_dashboards,
    plot_confidence_intervals,
    render_evaluation,
)

__all__ = [
    "RenderedEvaluation",
    "RenderedPlot",
    "get_dashboards",
    "plot_confidence_intervals",
    "render_evaluation",
]
