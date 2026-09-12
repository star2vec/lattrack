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


def chars_that_fit(ax, fontsize, frac=1.0):
    """How many characters of `fontsize` fit across `frac` of the axes width (sans-serif,
    ~0.55 em average advance)."""
    f = ax.figure
    w_in = ax.get_position().width * f.get_figwidth() * frac
    return max(12, int(w_in * 72 / (fontsize * 0.52)))


def wrap(ax, text, fontsize, frac=1.0):
    """Re-wrap `text` (existing newlines kept as paragraph breaks) to the axes width."""
    import textwrap
    n = chars_that_fit(ax, fontsize, frac)
    return "\n".join(textwrap.fill(part, n) if part else "" for part in text.split("\n"))


def title(ax, text, fontsize=8.5, pad=8, frac=1.0):
    """Left-aligned title wrapped to the axes width so it never hangs past the panel."""
    ax.set_title(wrap(ax, text, fontsize, frac), loc="left", fontsize=fontsize, pad=pad)


def audit(f):
    """Report text that hangs past the right edge of its axes or of the canvas. Titles and
    long notes are the usual offenders; with bbox='tight' they do not get clipped, they
    widen the canvas and leave the figure lopsided."""
    f.canvas.draw()
    W = f.get_figwidth() * f.dpi
    tol = 0.01 * W
    for ax in f.axes:
        ax_right = ax.get_window_extent().x1
        for t in ax.texts + [ax.title, ax.xaxis.label, ax._left_title, ax._right_title]:
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent()
            if bb.x1 > ax_right + tol or bb.x1 > W + tol:
                print(f"  OVERFLOW right: {t.get_text()[:50]!r} ends {bb.x1 - ax_right:+.0f}px past its axes"
                      + (f", {bb.x1 - W:+.0f}px past the canvas" if bb.x1 > W else ""))
    for lg in f.legends:
        bb = lg.get_window_extent()
        if bb.x1 > W + tol:
            print(f"  OVERFLOW right: figure legend {bb.x1 - W:+.0f}px past the canvas")


def save(f, name):
    """PNG for the post, SVG for editing; both under out/. Prints an overflow audit."""
    OUT.mkdir(exist_ok=True)
    audit(f)
    f.savefig(OUT / f"{name}.png")
    f.savefig(OUT / f"{name}.svg")
    print(f"wrote {OUT / name}.png / .svg")


def panel_label(ax, s, x=-0.06, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left")


def note(ax, s, x=0.01, y=0.01, **kw):
    """Small provenance/caption text inside an axes, bottom-left by default."""
    ax.text(x, y, s, transform=ax.transAxes, fontsize=7, color="#555555", va="bottom", ha="left", **kw)
