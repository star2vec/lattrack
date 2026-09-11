"""Figure 3: can the detector see a reversal when one is manufactured? Both models.

Donor thoughts are blended into the recipient at strength alpha (graph model: last
intermediate pass; Huginn: last prompt position from loop 20 on). Ground truth is the
model's own answer moving. Three rates per alpha: the answer moved; the detector fired
when it moved; the detector fired when it did not. A calibrated detector separates the
last two. Data: results/seed0/induced_rows.jsonl (RESULT 8) and
results/huginn_induced/summary.json (RESULT 9).
"""

import json
import sys

import numpy as np
from style import ANSWER, CONTROL, FULL, RESULTS, RIVAL, ROOT, fig, note, panel_label, save

sys.path.insert(0, str(ROOT / "src" / "lattrack"))
from stats import bootstrap  # noqa: E402


def graph_rates():
    rows = [json.loads(l) for l in open(RESULTS / "seed0" / "induced_rows.jsonl")]
    alphas = sorted(float(a) for a in rows[0]["alphas"])
    out = {}
    for a in alphas:
        R = [r["alphas"][str(a)] for r in rows]
        moved = [x for x in R if x["answer_changed"]]
        still = [x for x in R if not x["answer_changed"]]
        b = lambda v: bootstrap([float(x) for x in v], np.mean) if v else None
        out[a] = {"moved": b([x["answer_changed"] for x in R]),
                  "fired_moved": b([x["emb"]["real"]["crossed"] for x in moved]),
                  "fired_still": b([x["emb"]["real"]["crossed"] for x in still]),
                  "null": b([x["emb"]["null"]["crossed"] for x in R]), "n": len(R)}
    return out


def huginn_rates():
    s = json.load(open(RESULTS / "huginn_induced" / "summary.json"))
    out = {}
    for a, c in s["by_alpha"].items():
        out[float(a)] = {"moved": c["answer_changed"], "fired_moved": c["fires_when_answer_moved"],
                         "fired_still": c["fires_when_answer_did_not"], "null": c["null_fires"], "n": s["n_pairs"]}
    return out


def draw(ax, rates, title):
    xs = sorted(rates)
    def line(key, color, ls, label, marker="o"):
        pts = [(a, rates[a][key]) for a in xs if rates[a][key] is not None]
        if not pts:
            return
        x = [p[0] for p in pts]; y = [p[1]["point"] for p in pts]
        lo = [p[1]["point"] - p[1]["lo"] for p in pts]; hi = [p[1]["hi"] - p[1]["point"] for p in pts]
        ax.errorbar(x, y, yerr=[lo, hi], color=color, ls=ls, marker=marker, ms=4, lw=1.6, capsize=2, label=label)
    line("moved", "#444444", ":", "the answer actually moved", marker="s")
    line("fired_moved", ANSWER, "-", "detector fired — answer moved")
    line("fired_still", RIVAL, "-", "detector fired — answer did not move")
    line("null", CONTROL, "--", "arbitrary pair crossed", marker="")
    ax.set_xlabel("injection strength α"); ax.set_ylim(-0.03, 1.03); ax.set_title(title, loc="left", fontsize=9, pad=10)
    ax.set_xticks(xs)


g, h = graph_rates(), huginn_rates()
f = fig(FULL, 3.4)
ax1, ax2 = f.subplots(1, 2, sharey=True)
f.subplots_adjust(wspace=0.25)
draw(ax1, g, f"graph model, n={g[1.0]['n']} graphs\nthe detector separates")
draw(ax2, h, f"Huginn, n={h[1.0]['n']} question pairs\nit does not")
ax1.set_ylabel("rate")
r0 = h[0.0]["fired_still"]["point"]
ax2.annotate(f"fires with nothing injected: {r0:.1%}", xy=(0.0, r0), xytext=(0.08, 0.93),
             fontsize=8, color=RIVAL, arrowprops=dict(arrowstyle="-", color=RIVAL, lw=0.8))
panel_label(ax1, "A", x=-0.12, y=1.06); panel_label(ax2, "B", x=-0.06, y=1.06)
handles, labels = ax1.get_legend_handles_labels()
f.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
save(f, "fig3_calibration")
