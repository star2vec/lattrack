"""Figure 6: the checks, the models, and where each landed. Every cell names its RESULT in LOG.md.

Colours are about the INSTRUMENT, not the verdict: blue = the result stands on a check that
was itself validated on this model; amber = the result stands but the detector was not
(or could not be) calibrated on this model; red = the check failed; grey = not applicable;
hatched = not run. CODI is the third architecture family and is greyed with the reason.
"""

import re
import textwrap

from matplotlib.patches import Patch, Rectangle
from style import FULL, ROOT, fig, save

# the text model's id, read from the lens source rather than importing it (torch)
TEXT_MODEL = re.search(r'MODEL_ID\s*=\s*"([^"]+)"', (ROOT / "src" / "lattrack" / "wait_lens.py").read_text()).group(1).split("/")[-1]
W = 19   # wrap width inside a cell

COLS = ["leader changes\nabove an\narbitrary pair?", "at the noise floor\n(restarts,\nredraws)?",
        "detector catches\nan induced\nreversal?", "rival held but\nnever close?",
        "same verdict in\na second basis?", "readout precision\nruled out?"]
ROWS = [("2-layer graph model", "from scratch, feed-back · ProsQA"),
        ("Huginn-0125", "depth-recurrent, 3.5B · ARC"),
        (TEXT_MODEL, "explicit chain of thought · ARC, math traces"),
        ("CODI-GPT-2", "feed-back from a pretrained LM · GSM8K")]
S, U, F, N, X = "supports", "unbacked", "fail", "na", "notrun"
CELLS = [
    [("no — at the null rate\nR1", S), ("flips sit at the redraw floor\nR1", S), ("yes — 63–73% vs 8–17%\nR8", S),
     ("held late, 4–6 spreads behind\nR7", S), ("yes — causal-Jacobian basis\nR5", S), ("fp32 throughout", S)],
    [("no — at the null rate\nn=200 · R2", U), ("changes at the reseed floor\nR2", U), ("NO — 78% vs 83%;\n34.5% with nothing done · R9", F),
     ("n.a.", N), ("n.a.", N), ("yes — fp32 letter readout\nR12", S)],
    [("'wait' is not where the\nleaning moves · R3", S), ("phrasing noise exceeds\nwithin-trace change · R4", U), ("follows documented reversals,\n20× the control · R6", S),
     ("n.a.", N), ("n.a.", N), ("fp32 readout", S)],
    [("not run — no natural\ncandidate pair", X), ("not run", X), ("calibratable (pair by\nconstruction) — not run", X),
     ("n.a.", N), ("n.a.", N), ("—", N)],
]
FILL = {S: "#d9e4f0", U: "#f5e6bf", F: "#f1cfcd", N: "#f0f0f0", X: "#ffffff"}
LEGEND = [(S, "result stands, detector validated on this model"), (U, "result stands, detector not calibrated here"),
          (F, "check failed"), (N, "not applicable"), (X, "not run")]

f = fig(FULL, 4.8)
ax = f.add_axes([0.21, 0.15, 0.78, 0.70])
nr, nc = len(ROWS), len(COLS)
ax.set_xlim(0, nc); ax.set_ylim(nr, 0); ax.axis("off")
for j, col in enumerate(COLS):
    ax.text(j + 0.5, -0.06, col, ha="center", va="bottom", fontsize=6.6, fontweight="bold", linespacing=1.1)
for i, (name, sub) in enumerate(ROWS):
    grey = name.startswith("CODI")
    ax.text(-0.06, i + 0.38, name, ha="right", va="center", fontsize=7.5, fontweight="bold", color="#888888" if grey else "#222222")
    ax.text(-0.06, i + 0.66, textwrap.fill(sub, 26), ha="right", va="center", fontsize=6.2, color="#888888" if grey else "#555555")
    for j, (txt, st) in enumerate(CELLS[i]):
        ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, fc=FILL[st], ec="#cfcfcf", lw=0.6,
                               hatch="////" if st == X else None, alpha=0.75 if grey else 1.0))
        ax.text(j + 0.5, i + 0.5, "\n".join(textwrap.fill(part, W) for part in txt.split("\n")), ha="center", va="center",
                fontsize=6.2, linespacing=1.15, color="#999999" if (grey or st == N) else "#222222")
handles = [Patch(fc=FILL[k], ec="#cfcfcf", hatch="////" if k == X else None, label=lab) for k, lab in LEGEND]
f.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.58, 0.01))
save(f, "fig6_checklist")
print("text model:", TEXT_MODEL)
