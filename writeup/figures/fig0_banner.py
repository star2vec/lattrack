"""Opening banner: a scratch page of working, one attempt struck out, an arrow looping back
to an earlier line, a boxed answer. No legible text. Backtracking as it looks when it is
legible — the thing the latent trajectories in the figures never show.

3:1, content kept inside the middle half so a centred 1.9:1 social-card crop still holds
the strike-out, the loop-back and the box. Hand-drawn wobble on every stroke so it reads as
a page, not a diagram. Two variants (dense / sparse); pick one and it becomes fig0_banner.

    ../../.venv/bin/python fig0_banner.py [dense|sparse|both]
"""

import sys

import numpy as np
from style import ANSWER, INK, RIVAL, fig, save

PAPER, RULE = "#faf8f2", "#e9e5dc"
W, H = 7.2, 2.4                      # inches, 3:1
XMAX, YMAX = 3.0, 1.0


def wobble(xs, ys, rng, amp):
    """Slow random drift along a stroke: the hand, not the ruler."""
    n = len(xs)
    def drift():
        d = np.cumsum(rng.normal(0, 1, n))
        k = max(3, n // 12)
        d = np.convolve(d, np.ones(k) / k, mode="same")
        return amp * (d - d.mean()) / (np.abs(d).max() + 1e-9)
    return xs + drift(), ys + drift()


def word(ax, x0, x1, y, rng, color=INK, lw=None, amp=0.02):
    """One 'word': a cursive-looking stroke. Two frequencies, amplitude that swells and
    shrinks along the word, an ascender or descender or two, and a slight upward slant."""
    n = max(12, int((x1 - x0) * 500))
    xs = np.linspace(x0, x1, n)
    u = (xs - x0) / max(x1 - x0, 1e-6)
    env = 0.6 + 0.6 * np.abs(np.sin(np.pi * u * rng.uniform(1, 2.5) + rng.uniform(0, 3)))
    f1, f2 = rng.uniform(8, 13), rng.uniform(20, 34)
    ys = (y + rng.uniform(0.0, 0.012) * u                                 # slant
            + env * amp * np.sin(2 * np.pi * (xs - x0) * f1 + rng.uniform(0, 6))
            + 0.45 * env * amp * np.sin(2 * np.pi * (xs - x0) * f2 + rng.uniform(0, 6)))
    for _ in range(rng.integers(0, 3)):                                  # ascenders / descenders
        c = rng.uniform(x0 + 0.015, x1 - 0.015)
        ys += rng.choice([1.8, -1.5]) * amp * np.exp(-((xs - c) / 0.007) ** 2)
    xs, ys = wobble(xs, ys, rng, 0.004)
    ax.plot(xs, ys, color=color, lw=lw or rng.uniform(1.15, 1.45), solid_capstyle="round", zorder=3)


def line_of_working(ax, x0, x1, y, rng, n_words=None, **kw):
    """Words with gaps between them; returns the x where the line ends."""
    n_words = n_words or rng.integers(3, 6)
    cuts = np.sort(rng.uniform(x0 + 0.05, x1 - 0.05, n_words - 1))
    edges = [x0, *cuts, x1]
    for a, b in zip(edges[:-1], edges[1:]):
        gap = rng.uniform(0.02, 0.035)
        if b - gap - a > 0.04:
            word(ax, a, b - gap, y, rng, **kw)
    return x1


def strike(ax, x0, x1, y, rng):
    xs = np.linspace(x0 - 0.015, x1 + 0.015, 80)
    ys = np.linspace(y + 0.006, y - 0.012, 80)
    xs, ys = wobble(xs, ys, rng, 0.006)
    ax.plot(xs, ys, color=RIVAL, lw=1.7, solid_capstyle="round", zorder=4)


def loop_back(ax, x_from, y_from, x_to, y_to, rng, bulge=0.2):
    """A curve out into the left margin from a later line back up to an earlier one."""
    p0 = np.array([x_from, y_from]); p2 = np.array([x_to, y_to])
    p1 = np.array([min(x_from, x_to) - bulge, (y_from + y_to) / 2])
    t = np.linspace(0, 1, 120)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    xs, ys = wobble(pts[:, 0], pts[:, 1], rng, 0.006)
    ax.plot(xs, ys, color=RIVAL, lw=1.6, solid_capstyle="round", zorder=4)
    # arrowhead along the final tangent
    d = pts[-1] - pts[-6]; d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    for s in (+1, -1):
        tip = pts[-1]; back = tip - 0.045 * d + s * 0.028 * n
        ax.plot([tip[0], back[0]], [tip[1], back[1]], color=RIVAL, lw=1.6, solid_capstyle="round", zorder=4)


def box(ax, x0, x1, y, rng, pad=0.05):
    cx = [x0 - pad, x1 + pad, x1 + pad, x0 - pad, x0 - pad]
    cy = [y - pad * 0.9, y - pad * 0.9, y + pad * 1.1, y + pad * 1.1, y - pad * 0.9]
    xs = np.concatenate([np.linspace(cx[i], cx[i + 1], 40) for i in range(4)])
    ys = np.concatenate([np.linspace(cy[i], cy[i + 1], 40) for i in range(4)])
    xs, ys = wobble(xs, ys, rng, 0.005)
    ax.plot(np.append(xs, xs[0]), np.append(ys, ys[0]), color=ANSWER, lw=1.7, solid_capstyle="round", zorder=4)


RULES = np.arange(0.08, YMAX, 0.13)          # ruled lines; a row of working sits just above one


def page(ax):
    ax.set_xlim(0, XMAX); ax.set_ylim(0, YMAX); ax.axis("off")
    ax.figure.patch.set_facecolor(PAPER)
    ax.add_patch(__import__("matplotlib").patches.Rectangle((0, 0), XMAX, YMAX, fc=PAPER, ec="none", zorder=0))
    for y in RULES:
        ax.plot([0, XMAX], [y, y], color=RULE, lw=0.8, zorder=1)


def on_rule(i):
    return float(RULES[i] + 0.04)


def dense(seed=7):
    rng = np.random.default_rng(seed)
    f = fig(W, H); ax = f.add_axes([0, 0, 1, 1]); page(ax)
    x0, x1 = 0.95, 2.05
    rows = [on_rule(6), on_rule(5), on_rule(4), on_rule(3), on_rule(2), on_rule(0)]
    line_of_working(ax, x0, x1 - 0.1, rows[0], rng)
    line_of_working(ax, x0, x1, rows[1], rng)
    e = line_of_working(ax, x0, x1 - 0.25, rows[2], rng, n_words=3); strike(ax, x0, e, rows[2], rng)
    line_of_working(ax, x0 + 0.08, x1 - 0.05, rows[3], rng)
    line_of_working(ax, x0 + 0.08, x1 - 0.3, rows[4], rng, n_words=3)
    loop_back(ax, x0 + 0.05, rows[3] + 0.01, x0 - 0.03, rows[1] - 0.005, rng, bulge=0.22)
    bx0, bx1 = x0 + 0.45, x0 + 0.85
    word(ax, bx0, bx1, rows[5], rng, color=INK, lw=1.5); box(ax, bx0, bx1, rows[5], rng)
    return f


def sparse(seed=11):
    rng = np.random.default_rng(seed)
    f = fig(W, H); ax = f.add_axes([0, 0, 1, 1]); page(ax)
    x0, x1 = 1.0, 2.0
    rows = [on_rule(6), on_rule(5), on_rule(3), on_rule(2), on_rule(0)]
    line_of_working(ax, x0, x1 - 0.15, rows[0], rng, n_words=3)
    e = line_of_working(ax, x0, x1 - 0.3, rows[1], rng, n_words=3); strike(ax, x0, e, rows[1], rng)
    line_of_working(ax, x0 + 0.08, x1, rows[2], rng, n_words=4)
    line_of_working(ax, x0 + 0.08, x1 - 0.35, rows[3], rng, n_words=2)
    loop_back(ax, x0 + 0.05, rows[2] + 0.01, x0 - 0.03, rows[0] - 0.005, rng, bulge=0.2)
    bx0, bx1 = x0 + 0.4, x0 + 0.75
    word(ax, bx0, bx1, rows[4], rng, color=INK, lw=1.5); box(ax, bx0, bx1, rows[4], rng)
    return f


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("dense", "both"):
        save(dense(), "fig0_banner_dense")
    if which in ("sparse", "both"):
        save(sparse(), "fig0_banner_sparse")
