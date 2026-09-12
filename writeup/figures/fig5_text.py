"""Figure 5: the same kind of readout on a text model, where a change of mind is legible.

Left: a reasoning trace with three "wait"s, one of them a literal "Wait, no, actually";
the leaning toward the model's final answer, read under three forcing suffixes (faint)
and their mean (bold), does not move at any of them. Right: a downloaded trace in which
the model documentably switched its final answer; the leaning for (later answer vs
earlier answer) crosses zero at the written switch while a control pair of numbers stays
flat. Data: results/wait/wait_rows.jsonl (RESULT 3/4), results/validation/
validation_stated.jsonl (RESULT 6).
"""

import json
import re
import textwrap

import numpy as np
from style import ANSWER, CONTROL, FAINT, FULL, RESULTS, RIVAL, fig, panel_label, save, title, wrap

WAIT_QID = "Mercury_7007858"   # rule: every wait window has no 4-way leader change and a lean swing under the
                               # phrasing-noise q95 (9 of the 12 wait traces with a parsed answer qualify; this one
                               # has two waits and a literal step correction). Not "Mercury_SC_LBS10272": its third
                               # wait coincides with a 3.5-logit hardening of the answer it already had.
REV_IDX = 15                          # 2009 -> 4019, "I made a mistake" at token 1576 of 3000

f = fig(FULL, 5.0)
gs = f.add_gridspec(2, 2, height_ratios=[3.2, 0.9], hspace=0.40, wspace=0.34)
ax1, ax2 = f.add_subplot(gs[0, 0]), f.add_subplot(gs[0, 1])
tx1, tx2 = f.add_subplot(gs[1, 0]), f.add_subplot(gs[1, 1])
for t in (tx1, tx2):
    t.axis("off")

# ---- left: "wait" on a science question -------------------------------------------
rows = [json.loads(l) for l in open(RESULTS / "wait" / "wait_rows.jsonl")]
r = next(x for x in rows if x["qid"] == WAIT_QID)
noise = json.load(open(RESULTS / "wait" / "wait_summary.json"))["decoder_noise_abs_gap_change"]["q95"]
pi = "ABCD".index(r["parsed"])
ts = sorted(int(t) for t in r["reads"])


def lean(t, key):
    s = np.array(r["reads"][str(t)][key])
    return s[pi] - np.delete(s, pi).max()


curves = {k: np.array([lean(t, k) for t in ts]) for k in ("S0", "S1", "S2")}
for k, c in curves.items():
    ax1.plot(ts, c, color=ANSWER, alpha=0.28, lw=0.8)
ax1.plot(ts, np.mean(list(curves.values()), 0), color=ANSWER, lw=1.8, label="mean of three phrasings")
ax1.axhline(0, color=FAINT, lw=1, zorder=0)
for i, w in enumerate(r["waits"]):
    ax1.axvline(w, color=RIVAL, lw=1, ls="--")
    ax1.text(w, 10.0, "wait", color=RIVAL, fontsize=8, ha="center", va="bottom")
# noise scale bar
x0 = ts[-1] * 0.985
ax1.plot([x0, x0], [-1.0, -1.0 + noise], color=CONTROL, lw=2)
ax1.text(x0 - 6, -1.0 + noise / 2, f"phrasing\nnoise (q95)\n{noise:.1f}", color=CONTROL, fontsize=7, ha="right", va="center")
ax1.set_ylim(-3.5, 10.9); ax1.set_xlim(0, r["T"])
ax1.set_xlabel("token in the trace"); ax1.set_ylabel("leaning toward the final answer (logits)")
title(ax1, "a reasoning model says \"wait\" twice: the leaning does not move there", fontsize=9)
ax1.legend(loc="lower right", fontsize=7)
panel_label(ax1, "A", x=-0.22, y=1.08)

m = re.search(r"(?i)\bwait\b[^.]*\.[^.]*\.", r["trace"])
lo = r["trace"].rfind(". ", 0, m.start()) + 2
excerpt = r["trace"][lo:m.end()].replace("\n", " ")
tx1.text(0.0, 0.98, f"at the first \"wait\" (it ends on {r['parsed']}, correct {r['correct']}):", fontsize=7.2, color="#555555", va="top")
tx1.text(0.0, 0.76, wrap(tx1, "“…" + excerpt + "”", 7.5), fontsize=7.5, va="top", style="italic")

# ---- right: a documented reversal in a downloaded trace ------------------------
v = [json.loads(l) for l in open(RESULTS / "validation" / "validation_stated.jsonl")]
q = next(x for x in v if x["idx"] == REV_IDX)
S = np.array(q["scores"]); pos = np.array(q["positions"])
lean_r = S[:, 1] - S[:, 0]; null_r = S[:, 3] - S[:, 2]
ax2.plot(pos, lean_r, color=ANSWER, marker="o", ms=3, lw=1.6, label=f"later {q['to']} vs earlier {q['from']}")
ax2.plot(pos, null_r, color=CONTROL, marker="o", ms=3, lw=1.2, ls="--", label=f"control: {q['nulls'][1]} vs {q['nulls'][0]}")
ax2.axhline(0, color=FAINT, lw=1, zorder=0)
ax2.axvline(q["switch_tok"], color=RIVAL, lw=1, ls="--")
ax2.text(q["switch_tok"], 3.6, f"“{q['stated_phrase']}”", color=RIVAL, fontsize=8, ha="center", va="bottom")
ax2.text(820, -1.45, f"earlier answer ({q['from']}) preferred", fontsize=7, color=ANSWER, va="bottom")
ax2.text(pos[-1], 3.0, f"later answer ({q['to']}) preferred", fontsize=7, color=ANSWER, ha="right", va="bottom")
ax2.set_ylim(-6.8, 4.6); ax2.set_xlim(0, q["T"] + 40)
ax2.set_xlabel("token in the trace"); ax2.set_ylabel("log P(later answer) − log P(earlier answer)")
title(ax2, "a downloaded trace with a written change of answer: the leaning crosses at the switch", fontsize=9)
ax2.legend(loc="lower right", fontsize=7)
panel_label(ax2, "B", x=-0.22, y=1.08)
tx2.text(0.0, 0.95, wrap(tx2, f"reading stops at {q['T']:,} tokens; the trace continues to its final answer {q['to']}, "
         "the documented one. Grey: two other numbers from the same trace, read the same way "
         "(one early read is off-scale).", 7.5, frac=0.95), fontsize=7.5, color="#555555", va="top")
save(f, "fig5_text")
print("wait excerpt:", excerpt[:200])
