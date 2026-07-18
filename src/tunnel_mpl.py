"""Plot theme for the curriculum notebooks.

Restrained matplotlib defaults: a muted colour cycle, no chartjunk, and
consistent sizing so figures read the same across all thirteen modules.
Usage, once per notebook, immediately after the imports::

    from src.tunnel_mpl import apply_theme
    apply_theme()
"""

import matplotlib as mpl

# A muted, colour-blind-considerate cycle (Okabe–Ito, reordered).
_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
]


def apply_theme() -> None:
    """Apply the project matplotlib theme to the current session."""
    mpl.rcParams.update(
        {
            "figure.figsize": (7.0, 4.0),
            "figure.dpi": 110,
            "figure.autolayout": True,
            "axes.prop_cycle": mpl.cycler(color=_CYCLE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.titlesize": 11,
            "axes.titleweight": "medium",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "font.family": "sans-serif",
        }
    )
