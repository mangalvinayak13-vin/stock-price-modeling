"""
analysis/plot_style.py
========================
Shared dark-theme color palette and matplotlib styling applied consistently
across every chart in the project (dashboard + report figures), so the app
reads as one designed system instead of a patchwork of default matplotlib
colors that happen to clash with Streamlit's dark background.

Palette source: the validated reference categorical/status palette (dark
mode variant), confirmed via the project's palette validator to pass every
lightness/CVD-separation/normal-vision/contrast check against this exact
dark surface (#1a1a19) before use -- not eyeballed.

CONSISTENT ROLE MAPPING used everywhere a chart compares the same entities
(don't reassign these per-chart -- identity should stay attached to the
same color everywhere, per the "color follows the entity" rule):
    "actual / realized / empirical" data -> ink (TEXT_PRIMARY), solid, thick
    GBM                                  -> CAT[0] (blue)
    Heston                               -> CAT[1] (orange)
    Double Heston                        -> CAT[2] (aqua)
    Black-Scholes (shown standalone)     -> CAT[3] (yellow)
    "calm" regime                        -> CAT[0] (blue)
    "turbulent" regime                   -> CAT[1] (orange)
    pass / good status                   -> STATUS_GOOD
    fail / rejected / critical status    -> STATUS_CRITICAL
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Surfaces & ink (dark mode) ---
SURFACE = "#1a1a19"        # chart background
PAGE = "#0d0d0d"           # figure background (outside axes)
TEXT_PRIMARY = "#ffffff"
TEXT_SECONDARY = "#c3c2b7"
TEXT_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
AXIS = "#383835"

# --- Categorical palette (dark mode, validated order -- see module docstring) ---
CAT = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]

# --- Status palette (fixed, never reused for series identity) ---
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# Fixed role assignments (see docstring) -- import these names directly
# rather than indexing CAT[] by hand, so a chart can't accidentally swap
# which color means "Heston" vs "Double Heston" between two plots.
COLOR_ACTUAL = TEXT_PRIMARY
COLOR_GBM = CAT[0]
COLOR_HESTON = CAT[1]
COLOR_DOUBLE_HESTON = CAT[2]
COLOR_BLACK_SCHOLES = CAT[3]
COLOR_CALM = CAT[0]
COLOR_TURBULENT = CAT[1]


def apply_dark_style(fig, axes=None):
    """
    Apply the dark theme to a matplotlib figure (and its axes) in place:
    dark surfaces, light ink for text/ticks, muted gridlines, no heavy
    spine box (just left+bottom, in the muted axis color) -- the
    "recessive grid/axes, marks carry the color" rule from the skill.
    """
    fig.patch.set_facecolor(PAGE)
    if axes is None:
        axes = fig.get_axes()
    elif isinstance(axes, np.ndarray):
        # plt.subplots(nrows>1, ncols>1) returns a 2D array of Axes --
        # flatten it so we iterate individual Axes objects, not rows
        # (which are themselves 1D arrays and would break ax.set_facecolor).
        axes = axes.flatten()
    elif not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.title.set_color(TEXT_PRIMARY)
        ax.xaxis.label.set_color(TEXT_SECONDARY)
        ax.yaxis.label.set_color(TEXT_SECONDARY)
        ax.tick_params(colors=TEXT_MUTED, labelcolor=TEXT_SECONDARY)
        for spine_name in ["top", "right"]:
            ax.spines[spine_name].set_visible(False)
        for spine_name in ["left", "bottom"]:
            ax.spines[spine_name].set_color(AXIS)
        ax.grid(True, color=GRIDLINE, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor(SURFACE)
            legend.get_frame().set_edgecolor(AXIS)
            for text in legend.get_texts():
                text.set_color(TEXT_SECONDARY)
    return fig


def new_dark_fig(figsize=(9, 5), nrows=1, ncols=1, **kwargs):
    """Convenience: create a figure/axes pre-styled for the dark theme."""
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)
    apply_dark_style(fig, axes)
    return fig, axes
