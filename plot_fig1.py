"""Redraws Fig. 1 (int8 test accuracy against MACs per inference) from results/results_icct.json.

Usage:  python3 plot_fig1.py [results/results_icct.json] [fig1.png]

Points are means over 20 splits (UCI digits) or 5 seeds (MNIST, Fashion-MNIST);
error bars are 95% Student-t intervals of the mean. The grey line is the unpruned int8 baseline.
Retention is 75, 50, 35, 25 and 15% from right to left.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

src = sys.argv[1] if len(sys.argv) > 1 else "results/results_icct.json"
out = sys.argv[2] if len(sys.argv) > 2 else "fig1.png"
data = json.load(open(src))

PANELS = [("digits", "UCI digits (20 splits)", 20),
          ("mnist", "MNIST (5 seeds)", 5),
          ("fashion", "Fashion-MNIST (5 seeds)", 5)]
STYLES = [("both_l2prod", "both layers, L2-product", "#1f77b4", "o", "-"),
          ("both_random", "both layers, random", "#ff7f0e", "s", "--"),
          ("layer1_l2prod", "layer 1 only, L2-product", "#2ca02c", "^", ":")]


def kfmt(v):
    return f"{v / 1000:.1f}k" if v < 10000 else f"{v / 1000:.0f}k"


plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

fig, axes = plt.subplots(2, 2, figsize=(4.8, 4.6))
flat = list(axes.flat)

for ax, (key, title, n) in zip(flat, PANELS):
    rows = data[key]["rows"]
    tcrit = stats.t.ppf(0.975, n - 1)
    for var, label, color, marker, ls in STYLES:
        rs = sorted([r for r in rows if r["variant"] == var], key=lambda r: r["macs"])
        x = [r["macs"] for r in rs]
        y = [r["acc_i8_mean"] for r in rs]
        e = [tcrit * r["acc_i8_std"] / n ** 0.5 for r in rs]
        ax.errorbar(x, y, yerr=e, color=color, marker=marker, ls=ls, lw=1.0, ms=3.5,
                    capsize=2, elinewidth=0.8, label=label)
    ax.axhline(data[key]["baseline"]["acc_i8_mean"], color="gray", lw=0.8, label="unpruned int8")
    ax.set_xscale("log")
    ticks = sorted(r["macs"] for r in rows if r["variant"] == "both_l2prod")
    ax.set_xticks(ticks)
    ax.set_xticklabels([kfmt(t) for t in ticks])
    ax.minorticks_off()
    ax.set_title(title)
    ax.set_xlabel("MACs per inference")
    ax.set_ylabel("int8 test accuracy (%)")

# fourth cell: shared legend
leg_ax = flat[3]
leg_ax.axis("off")
handles, labels = flat[0].get_legend_handles_labels()
leg_ax.legend(handles, labels, loc="center", frameon=False, fontsize=7.5)

fig.tight_layout(pad=0.6)
fig.savefig(out, dpi=300)
print("saved", out)
