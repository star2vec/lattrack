"""Figure 1: the three stories a per-step readout can tell. Schematic, no data.

Two candidates read at every latent step. A REVERSAL: the leader switches and stays
switched. A GAIN RAMP: same leader throughout, the contrast grows. NOISE: two near-tied
lines that cross by accident, landing differently on every restart.
"""

import numpy as np
from style import ANSWER, CONTROL_LIGHT, FAINT, FULL, RIVAL, fig, panel_label, save

x = np.linspace(0, 1, 60)
f = fig(FULL, 2.6)
axes = f.subplots(1, 3, sharey=True)

# A. reversal
a = axes[0]
rival = 1.6 - 2.2 * (1 / (1 + np.exp(-(x - 0.5) * 14)))          # leads, then falls
answer = -1.2 + 2.6 * (1 / (1 + np.exp(-(x - 0.5) * 14)))        # trails, then leads
a.plot(x, rival, color=RIVAL, lw=2); a.plot(x, answer, color=ANSWER, lw=2)
a.axvline(0.5, color=FAINT, lw=1, zorder=0)
a.set_title("a reversal"); a.text(0.5, -1.75, "leader switches\nand stays switched", ha="center", fontsize=8, color="#555555")

# B. gain ramp
b = axes[1]
b.plot(x, 0.25 + 1.6 * x ** 1.3, color=ANSWER, lw=2)
b.plot(x, 0.05 - 0.9 * x ** 1.3, color=RIVAL, lw=2)
b.set_title("a gain ramp"); b.text(0.5, -1.75, "same leader,\ncontrast grows", ha="center", fontsize=8, color="#555555")

# C. noise: three restarts, near-tied, crossing in different places
c = axes[2]
rng = np.random.default_rng(3)
for k, alpha in enumerate((1.0, 0.45, 0.45)):
    w1 = np.convolve(rng.normal(0, 1, 80), np.ones(12) / 12, mode="same")[10:70] * 0.9
    w2 = np.convolve(rng.normal(0, 1, 80), np.ones(12) / 12, mode="same")[10:70] * 0.9
    c.plot(x, 0.15 + w1, color=ANSWER, lw=1.6 if k == 0 else 1.1, alpha=alpha)
    c.plot(x, -0.15 + w2, color=RIVAL, lw=1.6 if k == 0 else 1.1, alpha=alpha)
c.set_title("noise"); c.text(0.5, -1.75, "near-tied lines cross by accident,\ndifferently on each restart", ha="center", fontsize=8, color="#555555")

for ax, lab in zip(axes, "ABC"):
    ax.set_ylim(-2.0, 2.1); ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    ax.spines["left"].set_visible(False); ax.set_xlabel("latent step →", fontsize=8)
    panel_label(ax, lab)
axes[0].set_ylabel("readout for each candidate")
axes[0].plot([], [], color=ANSWER, lw=2, label="the answer the model ends on")
axes[0].plot([], [], color=RIVAL, lw=2, label="the rival it is compared against")
f.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.12))
save(f, "fig1_stories")
