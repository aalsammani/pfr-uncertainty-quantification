"""Publication figure style and export.

Colour choices follow a colour-blind-validated categorical palette (fixed slot
order, never cycled); every categorical distinction is also carried by line
style and/or marker so figures remain legible in greyscale.  Magnitudes use a
single-hue sequential map.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .paths import FIGURES

__all__ = [
    "COLORS",
    "LINESTYLES",
    "MARKERS",
    "SEQ_CMAP",
    "INK",
    "MUTED",
    "set_style",
    "panel_label",
    "save_figure",
    "single_column",
    "double_column",
]

#: Categorical palette, fixed slot order (blue, orange, aqua, yellow, magenta, green, violet, red).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1)), (0, (1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

#: Single-hue sequential colour map (light -> dark blue).
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f2f6fc", "#a9c8ef", "#2a78d6", "#123a6b"])

#: Figure widths in inches (typical two-column journal layout).
single_column = 3.5
double_column = 7.2


def set_style() -> None:
    """Apply a consistent, restrained matplotlib style for all figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.linewidth": 0.6,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.prop_cycle": mpl.cycler(color=COLORS),
        }
    )


def panel_label(ax, label: str, x: float = -0.16, y: float = 1.04) -> None:
    """Place a bold panel label such as '(a)' at the upper-left corner of an axis."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color=INK)


def save_figure(fig, name: str, directory: Path | None = None, png_dpi: int = 400) -> dict:
    """Save a figure as vector PDF and high-resolution PNG; return the written paths.

    The PDF is embedded in the manuscript; the PNG is used for previews in the
    repository.  Metadata timestamps are removed so that re-running the
    notebooks produces byte-stable PDFs where the backend allows it.
    """
    directory = Path(directory) if directory is not None else FIGURES
    directory.mkdir(parents=True, exist_ok=True)
    pdf = directory / f"{name}.pdf"
    png = directory / f"{name}.png"
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": "pfr_uq", "Producer": None})
    fig.savefig(png, dpi=png_dpi, metadata={"Software": None})
    return {"pdf": str(pdf.relative_to(directory.parent.parent)) if directory == FIGURES else str(pdf),
            "png": str(png.relative_to(directory.parent.parent)) if directory == FIGURES else str(png)}


def log_ticks_clean(ax, axis: str = "both") -> None:
    """Use plain 10^k labels on log axes."""
    fmt = mpl.ticker.LogFormatterMathtext()
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def slope_triangle(ax, x0, y0, width_decades, slope, label=None, below=True, color=MUTED):
    """Draw a reference slope triangle on a log-log axis starting at (x0, y0)."""
    x1 = x0 * 10**width_decades
    y1 = y0 * (x1 / x0) ** slope
    if below:
        xs, ys = [x0, x1, x1, x0], [y0, y0, y1, y0] if slope > 0 else [y0, y1, y1, y0]
        xs = [x0, x1, x1, x0]
        ys = [y0, y0, y1, y0]
    else:
        xs = [x0, x0, x1, x0]
        ys = [y0, y1, y1, y0]
    ax.plot(xs, ys, color=color, lw=0.7)
    if label:
        ax.text(x1 * 1.08, np.sqrt(y0 * y1), label, color=color, fontsize=7, va="center", ha="left")
