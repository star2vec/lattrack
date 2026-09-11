"""One look for all figures in the post. Import this first in every fig*.py.

Widths are for a ~720 px post column: FULL for a one-figure row, HALF for side-by-side
panels drawn in one figure. Colours are named by ROLE, not by hue, so a reader who has
seen one figure can read the next: the answer the model ends on, the rival it is compared
against, an arbitrary/control pair, and greyed structure.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"

FULL = 7.2      # inches
TALL = 4.2

# roles
ANSWER = "#1f4e79"      # the answer the model ends on (target / correct / later answer)
RIVAL = "#c0504d"       # the tracked rival (decoy / top distractor / earlier answer)
CONTROL = "#8c8c8c"     # arbitrary or control pair
CONTROL_LIGHT = "#c8c8c8"
FRONTIER = "#f0a30a"    # reachable-but-not-candidate structure
FAINT = "#d9d9d9"
INK = "#222222"
OPTIONS = ["#1f4e79", "#c0504d", "#5b9b57", "#8e6bb3"]   # A B C D when four options are drawn
RESTART_ALPHAS = (1.0, 0.45, 0.45)                       # base, two re-inits

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "text.color": INK, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "font.family": "sans-serif",
})


def fig(width=FULL, height=TALL, **kw):
    return plt.figure(figsize=(width, height), **kw)


def save(f, name):
    """PNG for the post, SVG for editing; both under out/."""
    OUT.mkdir(exist_ok=True)
    f.savefig(OUT / f"{name}.png")
    f.savefig(OUT / f"{name}.svg")
    print(f"wrote {OUT / name}.png / .svg")


def panel_label(ax, s, x=-0.06, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def note(ax, s, x=0.01, y=0.01, **kw):
    """Small provenance/caption text inside an axes, bottom-left by default."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7, color="#555555", va="bottom", ha="left", **kw)
