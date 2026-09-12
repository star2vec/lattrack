"""Figure 2: one graph, read at every position — and the crossing rate over all of them.

A. Heatmap of every node in one test graph (rows) across the readout positions (columns:
the root thought, the latent thoughts, the answer position). Colour is the node's score
z-scored against the unreachable non-candidates at that position (RESULT 7's reference
class), so the frontier lighting up, the target pulling ahead, and the decoy being held
but never close are all one picture. Outlined cell = the top node at that position.
B. Crossing rate per transition for the tracked (target, decoy) pair against an arbitrary
pair and a matched-depth pair, all 419 test graphs, seed0 (RESULT 1; K=4 graphs shown).

Example rule (IDS.md): first graph in index order with K=4, answered correctly, exactly one
leader change and it D->T at a latent transition, the decoy still in the top three nodes
at the last latent position (so "held" is visible), and at least ten unreachable
non-candidates for the reference class.
"""

import json
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from style import ANSWER, CONTROL, CONTROL_LIGHT, FULL, RESULTS, RIVAL, ROOT, fig, panel_label, save, title, wrap

sys.path.insert(0, str(ROOT / "src" / "lattrack"))
from prompts import Prompt, pin_seed  # noqa: E402
from sets import TEST_OFFSET, load_test  # noqa: E402
from superposition import classify  # noqa: E402

RUN = "seed0"
rows = {r["gi"]: r for r in (json.loads(l) for l in open(RESULTS / RUN / "lens_rows_all.jsonl")) if r["variant"] == "base"}
test = load_test()


def pick():
    for gi in sorted(rows):
        r = rows[gi]
        if r["K"] != 4 or not r["correct"] or len(r["flips"]) != 1 or r["flips"][0]["dir"] != "D>T":
            continue
        if r["flips"][0]["t"] >= len(r["positions"]) - 2:      # the change must be at a latent transition, not into A
            continue
        if r["per_position"][-2]["rank_d"] > 3:
            continue
        pr = Prompt.from_sample(test[gi], pin_seed(gi + TEST_OFFSET, 0))
        if len(classify(pr)[1]) >= 10:
            return gi, r, pr
    raise SystemExit("no graph satisfies the rule")


gi, r, pr = pick()
depths, unreach, reach = classify(pr)
present = sorted(pr.nodes())
root = min(depths, key=depths.get)
order = sorted(present, key=lambda v: (0, depths[v]) if v in depths else (1, v))
labels = []
for v in order:
    tag = " root" if v == root else (" target" if v == pr.target else (" decoy" if v == pr.decoy else ""))
    labels.append(f"{v}{tag}" if tag else str(v))
Z, top = [], []
for per in r["per_position"]:
    s = np.array(per["node_logits"])
    ref = s[unreach]; mu, sd = ref.mean(), ref.std() + 1e-6
    Z.append([(s[v] - mu) / sd for v in order])
    top.append(order.index(present[int(np.argmax(s[present]))]))
Z = np.array(Z).T                                        # (nodes, positions)
names = {"root": "root\nthought", "A": "answer\nposition"}
cols = [names.get(p, f"latent\n{p}") for p in r["positions"]]

f = fig(FULL, 4.6)
gs = f.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.5)
ax, bx = f.add_subplot(gs[0]), f.add_subplot(gs[1])

cmap = LinearSegmentedColormap.from_list("post", ["#f4f4f4", "#c9d7e6", ANSWER])
im = ax.imshow(Z, aspect="auto", cmap=cmap, vmin=-1.5, vmax=8, interpolation="nearest")
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=7)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7.5)
for i, v in enumerate(order):
    if v in (pr.target, pr.decoy):
        ax.get_yticklabels()[i].set_color(ANSWER if v == pr.target else RIVAL)
        ax.get_yticklabels()[i].set_fontweight("bold")
for k, t in enumerate(top):
    ax.add_patch(Rectangle((k - 0.5, t - 0.5), 1, 1, fill=False, ec="#222222", lw=1.2))
n_reach = len([v for v in order if v in depths])
ax.axhline(n_reach - 0.5, color="#222222", lw=0.8)
box = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
ax.text(len(cols) - 0.6, 0.2, "reachable from the root", fontsize=6.5, va="top", ha="right", bbox=box)
ax.text(len(cols) - 0.6, n_reach + 0.2, "unreachable (reference class)", fontsize=6.5, va="top", ha="right", color="#555555", bbox=box)
cb = f.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.16, shrink=1.0)
cb.set_label(f"score, in spreads of the unreachable class\n(decoy held at +{Z[order.index(pr.decoy), -2]:.1f} late; target ends {Z[order.index(pr.target), -1] - Z[order.index(pr.decoy), -1]:.0f} clear)", fontsize=6.5); cb.ax.tick_params(labelsize=7)
for k in range(len(cols)):
    zt, zd = Z[order.index(pr.target), k], Z[order.index(pr.decoy), k]
    ax.text(k, -0.9, f"T{zt:+.1f}\nD{zd:+.1f}", ha="center", va="bottom", fontsize=6, color="#444444")
ax.set_ylim(len(order) - 0.5, -2.6)
title(ax, f"one graph (test #{gi}, depth {r['K']}): the leader changes once, at a latent step")

panel_label(ax, "A", x=-0.28, y=1.05)

# B. crossing rate per transition
s = json.load(open(RESULTS / RUN / "lens_summary_all.json"))["crossing_rate_forward_names"]
trans = ["root>l0", "l0>l1", "l1>l2", "l2>A"]
x = np.arange(len(trans))
for key, color, ls, label, dx in (("target_decoy", ANSWER, "-", "tracked pair: target vs decoy", -0.08),
                                  ("null_any", CONTROL, "--", "an arbitrary pair of nodes", 0.0),
                                  ("null_matched", CONTROL_LIGHT, "--", "an arbitrary pair, matched depth", 0.08)):
    pts = [s[key][t] for t in trans]
    y = [p["point"] for p in pts]; lo = [p["point"] - p["lo"] for p in pts]; hi = [p["hi"] - p["point"] for p in pts]
    bx.errorbar(x + dx, y, yerr=[lo, hi], color=color, ls=ls, marker="o", ms=4, lw=1.6, capsize=2, label=label)
bx.set_xticks(x); bx.set_xticklabels(["root→l0", "l0→l1", "l1→l2", "l2→answer"], fontsize=7.5)
bx.set_ylim(0, 0.6); bx.set_ylabel("fraction of graphs whose leader changes at this step", fontsize=8)
title(bx, f"all depth-4 test graphs (n={s['target_decoy']['l1>l2']['n']}): the tracked pair crosses no more than an arbitrary one")
bx.legend(loc="upper right", fontsize=7)
bx.annotate("the answer step\nalmost never\nchanges the leader", xy=(3, s["target_decoy"]["l2>A"]["point"]), xytext=(1.55, 0.09),
            fontsize=7, color=ANSWER, arrowprops=dict(arrowstyle="-", color=ANSWER, lw=0.8))
panel_label(bx, "B", x=-0.22, y=1.05)
save(f, "fig2_graph")
print(f"example graph gi={gi}, target {pr.target}, decoy {pr.decoy}, root {root}, leaders {r['leaders']}")
