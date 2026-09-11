"""Figure 4: Huginn — restarts on one question, and every question's settling over 64 loops.

A. One ARC-Easy question read at every recurrent loop under three random initial states
(the model's own randomised start). The four option probabilities wobble across each
other early, in different places each time, then settle on the same letter. Over all 50
questions the three restarts end on the same letter 96% of the time but share the exact
set of change-loops only 8% of the time (RESULT 2 reseed control).
B. Every question at K=64 as a strip over loops: shaded where the current leader differs
from the final one, sorted by the loop of the last change. K=16 and K=30 are literally
prefixes of this run (RESULT 11), marked as vertical lines with the "fraction of the
trajectory" the same last change would read as. Below: mean margin between the top two
options, which stops growing where the changes stop.

Example rule (IDS.md): a K=30 question with all three restarts on the correct final letter,
3-8 changes in each restart, all changes before loop 20, chosen as the one with the
LEAST overlap of change-loops between restarts (so the reseed scatter is visible).
"""

import collections
import json

import numpy as np
from matplotlib.colors import ListedColormap
from style import ANSWER, CONTROL, FAINT, FULL, OPTIONS, RESULTS, RIVAL, fig, panel_label, save

# ---- A: restarts on one question -------------------------------------------------------
by = collections.defaultdict(dict)
for l in open(RESULTS / "huginn_easy" / "lens_rows.jsonl"):
    r = json.loads(l); by[r["qid"]][r["condition"]] = r
CONDS = ("base", "init_1", "init_2")


def changes(row):
    lead = np.array(row["steps"]).argmax(1)
    return set(i for i in range(1, len(lead)) if lead[i] != lead[i - 1])


best = None
for q, c in by.items():
    if not all(k in c for k in CONDS):
        continue
    ch = [changes(c[k]) for k in CONDS]
    finals = {"ABCD"[np.array(c[k]["steps"])[-1].argmax()] for k in CONDS}
    if finals != {c["base"]["correct_letter"]} or not all(3 <= len(x) <= 8 and max(x) < 20 for x in ch):
        continue
    u = ch[0] | ch[1] | ch[2]
    overlap = len(ch[0] & ch[1] & ch[2]) / len(u)
    if best is None or overlap < best[0]:
        best = (overlap, q)
QID = best[1]
c = by[QID]
K = len(c["base"]["steps"])

f = fig(FULL, 5.4)
gs = f.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[4, 1.1], hspace=0.14, wspace=0.34)
ax = f.add_subplot(gs[:, 0]); hx = f.add_subplot(gs[0, 1]); mx = f.add_subplot(gs[1, 1], sharex=hx)

loops = np.arange(1, K + 1)
onset = next(i for i, v in enumerate(c["base"]["argmax_is_letter"]) if v) + 1      # first loop where a letter is the next token
ax.axvspan(0.5, onset - 0.5, color="#f1f1f1", zorder=0)
ax.text(onset - 0.7, 0.985, "letters not yet\nthe next token", fontsize=6.5, ha="right", va="top", color="#777777")
for j, cond in enumerate(CONDS):
    P = np.array(c[cond]["derived"]["probs"])
    for o in range(4):
        ax.plot(loops, P[:, o], color=OPTIONS[o], lw=1.8 if j == 0 else 1.0, alpha=1.0 if j == 0 else 0.45,
                ls="-" if j == 0 else "--", label=f"option {'ABCD'[o]}" + (" (correct)" if "ABCD"[o] == c["base"]["correct_letter"] else "") if j == 0 else None)
ax.plot([], [], color="#444444", lw=1.8, label="base run"); ax.plot([], [], color="#444444", lw=1.0, ls="--", alpha=0.6, label="two other initial states")
ax.set_xlim(1, K); ax.set_ylim(0, 1); ax.set_xlabel("recurrent loop"); ax.set_ylabel("probability among the four options")
ax.set_title("one question, three random initial states\nan early wobble, then the same letter", loc="left", fontsize=8.5, pad=8)
ax.legend(loc="upper right", fontsize=7)
ax.text(0.0, -0.13, "all 50 questions: the restarts end on the same letter 96% of the time and share the exact\nset of change-loops 8% of the time; before the shaded edge the four letters carry <5% of the\nnext-token mass, so the swings there are ratios of near-zero numbers",
        transform=ax.transAxes, fontsize=6.8, color="#555555", va="top")
panel_label(ax, "A", x=-0.2, y=1.05)

# ---- B: every question at K=64, sorted by last change ----------------------------------
rows = [json.loads(l) for l in open(RESULTS / "huginn_easy_K64" / "lens_rows.jsonl")]
rows = [r for r in rows if r["condition"] == "base"]
S = [np.array(r["steps"]) for r in rows]
K64 = len(S[0])
lead = np.array([s.argmax(1) for s in S])                       # (n, K64)
diff = (lead != lead[:, -1:]).astype(float)                      # 1 where the leader differs from the final one
last = [max([i for i in range(1, K64) if lead[q, i] != lead[q, i - 1]] or [0]) for q in range(len(rows))]
order = np.argsort(last)
hx.imshow(diff[order], aspect="auto", cmap=ListedColormap([FAINT, RIVAL]), interpolation="nearest", extent=(0.5, K64 + 0.5, len(rows) - 0.5, -0.5))
hx.set_yticks([]); hx.set_ylabel(f"{len(rows)} questions, sorted by their last change")
hx.set_title("all 50 questions over 64 loops; red = leader still differs from the final one\nK=16 and K=30 are prefixes of this run, so one last change reads as 0.62, 0.39 or 0.24 'of the way'",
             loc="left", fontsize=8, pad=8)
fr = {16: 0.62, 30: 0.39, 64: 0.24}                             # mean last-change loop / (K-1), RESULT 11
for k in (16, 30):
    hx.axvline(k + 0.5, color="#222222", lw=0.9, ls=":")
    hx.text(k + 1.0, -0.8, f"K={k}: {fr[k]:.2f}", fontsize=6.5, ha="left", va="bottom", color="#222222")
hx.text(K64, -0.8, f"K=64: {fr[64]:.2f}", fontsize=6.5, ha="right", va="bottom", color="#222222")
hx.set_ylim(len(rows) - 0.5, -2.6)
hx.text(K64 * 0.62, len(rows) * 0.55, "after ~loop 24: inert\n(1–3% of questions change per loop)", fontsize=7, color="#555555", ha="center")
panel_label(hx, "B", x=-0.06, y=1.05)

onset64 = int(np.median([next(i for i, v in enumerate(r["argmax_is_letter"]) if v) for r in rows])) + 1
hx.axvspan(0.5, onset64 - 0.5, color="white", alpha=0.55, zorder=2)
mx.axvspan(0.5, onset64 - 0.5, color="#f1f1f1", zorder=0)
srt = np.sort(np.array(S), axis=2)                               # (n, K64, 4)
margin = (srt[:, :, -1] - srt[:, :, -2]).mean(0)
mx.plot(np.arange(1, K64 + 1), margin, color=ANSWER, lw=1.6)
mx.set_ylim(0, max(margin) * 1.25); mx.set_ylabel("mean margin\n(logits)", fontsize=7.5)
mx.set_xlabel("recurrent loop"); mx.set_xlim(0.5, K64 + 0.5)
for k in (16, 30):
    mx.axvline(k + 0.5, color="#222222", lw=0.9, ls=":")
mx.text(K64 - 1, margin[-1] * 1.08, "top option minus runner-up:\ngrows to ~loop 16, then flat", fontsize=7, color=ANSWER, ha="right", va="bottom")
save(f, "fig4_huginn")
print(f"example question {QID}, correct {c['base']['correct_letter']}, overlap {best[0]:.2f}; K64 n={len(rows)}")
