#!/usr/bin/env python3
"""
Figure 1 of the paper: oracle coverage (best-of-N), chopthin vs. systematic resampling,
for all 15 model x benchmark cells. Three model panels side by side (dumbbell per cell:
open marker = systematic, filled = chopthin; teal = higher, red = lower, gray = tie).

The oracle counts are embedded below and are computed straight from the run outputs
(the per-question grades / oracle_all.py). Only matplotlib is required.

    python3 figures/figure1_oracle_ceiling.py    ->  oracle_ceiling_wide.pdf / .png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# (model, [(benchmark, systematic_oracle_count, chopthin_oracle_count, n), ...])
DATA = [
    ("Qwen2.5-Math-7B", [
        ("MATH500",    423, 435,  500),
        ("GSM8K",     1245, 1251, 1319),
        ("GPQA",       100, 108,  198),
        ("HumanEval",   81,  96,  164),
        ("AIME",        16,  20,   90),
    ]),
    ("Qwen2.5-7B", [
        ("MATH500",    406, 415,  500),
        ("GSM8K",     1241, 1250, 1319),
        ("GPQA",       100,  97,  198),
        ("HumanEval",   88,  90,  164),
        ("AIME",        37,  37,  270),
    ]),
    ("Qwen3-4B", [
        ("MATH500",    419, 424,  500),
        ("GSM8K",     1236, 1239, 1319),
        ("GPQA",        79,  90,  198),
        ("HumanEval",  119, 117,  164),
        ("AIME",        16,  18,   90),
    ]),
]

plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.linewidth": 0.8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C_SYS = "#9aa0a6"   # systematic (open marker)
C_UP  = "#0b7285"   # chopthin higher (teal)
C_DN  = "#c0392b"   # chopthin lower (red)
C_TIE = "#9aa0a6"   # tie (gray)

fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.85), sharey=True)

for ax, (model, cells) in zip(axes, DATA):
    labels = [c[0] for c in cells]
    ys = list(range(len(cells)))[::-1]          # MATH500 at top
    for yy, (name, sc, cc, n) in zip(ys, cells):
        sysv, chopv = 100.0 * sc / n, 100.0 * cc / n
        d = chopv - sysv
        col = C_UP if d > 0.05 else (C_DN if d < -0.05 else C_TIE)
        ax.plot([sysv, chopv], [yy, yy], color=col, lw=2.4, solid_capstyle="round", zorder=1)
        ax.scatter([sysv],  [yy], s=42, facecolors="white", edgecolors=C_SYS, linewidths=1.6, zorder=3)
        ax.scatter([chopv], [yy], s=50, color=col, zorder=4)
        ax.text(max(sysv, chopv) + 2.0, yy, f"{d:+.1f}", ha="left", va="center",
                fontsize=8.6, color=col)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels[::-1], fontsize=9.8)
    ax.set_title(model, fontsize=10.8, fontweight="bold", pad=6)
    ax.set_xlim(5, 110)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", ls=":", lw=0.6, color="0.85", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False)

fig.supxlabel("Oracle coverage: best-of-32 (%)", fontsize=10.5, y=0.02)

handles = [
    Line2D([0], [0], marker="o", mfc="white", mec=C_SYS, mew=1.6, ls="None", ms=7,
           label="Power-SMC (systematic)"),
    Line2D([0], [0], marker="o", color=C_UP, ls="None", ms=7.5, label="chopthin (ours)"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9.6,
           handletextpad=0.4, bbox_to_anchor=(0.5, 1.06))

fig.tight_layout(rect=[0, 0.04, 1, 0.99])
fig.savefig("oracle_ceiling_wide.pdf", bbox_inches="tight")
fig.savefig("oracle_ceiling_wide.png", dpi=200, bbox_inches="tight")
print("saved oracle_ceiling_wide.pdf / .png")
