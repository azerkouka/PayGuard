"""Static plots written next to the model."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
SERIES = "#2a78d6"


def _axes(title: str, subtitle: str, figsize: tuple[float, float]):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    height = figsize[1]
    fig.text(0.02, 1 - 0.3 / height, title, fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.6 / height, subtitle, fontsize=10, color=INK_SECONDARY)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    return fig, ax


def _save(fig, path: Path) -> None:
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.8 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_cost(
    thresholds: np.ndarray, costs: np.ndarray, threshold: float, path: Path
) -> None:
    best = costs.min()
    fig, ax = _axes(
        "Cost of each decision threshold",
        "Validation set, thousand TND. Too many blocked customers on the left, missed frauds "
        "on the right.",
        (8.5, 4.5),
    )
    ax.plot(thresholds, costs / 1000, color=SERIES, linewidth=2)
    ax.scatter(
        [threshold], [best / 1000], s=50, color=SERIES, edgecolor=SURFACE, linewidth=2, zorder=3
    )
    ax.annotate(
        f"minimum at {threshold:.3f}\n{best / 1000:,.1f}k TND",
        (threshold, best / 1000),
        xytext=(0, 28),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=INK,
    )
    ax.set_xlabel("Threshold (flag when score ≥ threshold)", color=INK_SECONDARY, fontsize=9)
    ax.set_ylim(0, 4 * best / 1000)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _save(fig, path)


def plot_shap_importance(importance: pd.Series, path: Path) -> None:
    """Horizontal bars, most important feature on top."""
    values = importance.sort_values()
    fig, ax = _axes(
        "What drives the fraud score",
        "Mean absolute SHAP value per feature, test set sample (log-odds)",
        (8, 0.32 * len(values) + 1.4),
    )
    bars = ax.barh(values.index, values.to_numpy(), height=0.6, color=SERIES)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8, color=INK_SECONDARY)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.margins(x=0.1, y=0.02)
    _save(fig, path)
