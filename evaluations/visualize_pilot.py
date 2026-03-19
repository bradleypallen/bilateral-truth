#!/usr/bin/env python3
"""
Pilot visualisation: performance table + bilateral truth value distribution
bar chart for the three-model TruthfulQA pilot (N=50, n=3 variants).
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), "results")

MODELS = [
    ("Claude Opus 4.1", "truthfulqa_complete_claude-opus-4-1-20250805_n3_modal_results.json"),
    ("GPT-4.1",         "truthfulqa_complete_gpt-4.1-2025-04-14_n3_modal_results.json"),
    ("DeepSeek-V3",     "truthfulqa_complete_deepseek_deepseek-chat_n3_modal_results.json"),
]

MODES = [
    ("null",  "VM(s\u2080, p)",   "Null situation"),
    ("src",   "VM(s_src, p)",               "Source situation"),
    ("modal", "\u25a1p",           "Modal"),
]

# BBL truth-value display order and colours  (\u27e8 = ⟨, \u27e9 = ⟩)
TV_ORDER  = ["<t,f>", "<f,t>", "<f,f>", "<t,t>", "<e,t>", "<t,e>", "<e,f>", "<f,e>", "<e,e>"]
TV_LABELS = {
    "<t,f>": "\u27e8t,f\u27e9",
    "<f,t>": "\u27e8f,t\u27e9",
    "<f,f>": "\u27e8f,f\u27e9",
    "<t,t>": "\u27e8t,t\u27e9",
    "<e,t>": "\u27e8e,t\u27e9",
    "<t,e>": "\u27e8t,e\u27e9",
    "<e,f>": "\u27e8e,f\u27e9",
    "<f,e>": "\u27e8f,e\u27e9",
    "<e,e>": "\u27e8e,e\u27e9",
}
TV_COLORS = {
    "<t,f>": "#2196F3",   # blue   – verified
    "<f,t>": "#F44336",   # red    – refuted
    "<f,f>": "#9E9E9E",   # grey   – ignorance
    "<t,t>": "#FF9800",   # orange – contradiction
    "<e,t>": "#CE93D8",
    "<t,e>": "#90CAF9",
    "<e,f>": "#FFCC80",
    "<f,e>": "#EF9A9A",
    "<e,e>": "#CFD8DC",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

data = {}
for label, fname in MODELS:
    with open(os.path.join(BASE, fname)) as f:
        data[label] = json.load(f)

N = data[list(data.keys())[0]]["total_samples"]

# ─────────────────────────────────────────────────────────────────────────────
# Figure layout: 2 rows
#   Row 0: performance table (rendered as a matplotlib table)
#   Row 1: stacked bar chart (3 sub-panels, one per mode)
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(
    2, 1,
    height_ratios=[1, 2.2],
    hspace=0.45,
)

# ── Panel 0: performance table ───────────────────────────────────────────────

ax_table = fig.add_subplot(gs[0])
ax_table.axis("off")

col_labels = [
    "Model",
    "VM(s\u2080, p)\nF1 / Cov",
    "VM(s_src, p)\nF1 / Cov",
    "\u25a1p\nF1 / Cov",
    "\u0394 F1\n(null\u2192modal)",
    "\u0394 Cov\n(null\u2192modal)",
]

rows = []
for label, _ in MODELS:
    r = data[label]
    row = [label]
    for mode_key, _, _ in MODES:
        f1  = r[f"{mode_key}_f1_macro"]
        cov = r[f"{mode_key}_coverage"]
        row.append(f"{f1:.3f} / {cov:.3f}")
    delta_f1  = r["modal_f1_macro"]  - r["null_f1_macro"]
    delta_cov = r["modal_coverage"]  - r["null_coverage"]
    row.append(f"{delta_f1:+.3f}")
    row.append(f"{delta_cov:+.3f}")
    rows.append(row)

tbl = ax_table.table(
    cellText=rows,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)

# Style header
for j in range(len(col_labels)):
    cell = tbl[0, j]
    cell.set_facecolor("#1565C0")
    cell.set_text_props(color="white", fontweight="bold")
    cell.set_height(0.28)

# Style data rows
row_colors = ["#E3F2FD", "#FAFAFA"]
for i, row in enumerate(rows):
    for j in range(len(col_labels)):
        cell = tbl[i + 1, j]
        cell.set_facecolor(row_colors[i % 2])
        cell.set_height(0.22)
        if j == 0:
            cell.set_text_props(fontweight="bold")
        # Highlight Δ columns
        if j >= 4:
            val = float(rows[i][j])
            cell.set_facecolor("#C8E6C9" if val >= 0 else "#FFCDD2")

ax_table.set_title(
    f"Table 1. TruthfulQA Pilot Results — BBL Modal Evaluation  "
    f"($N={N}$, $n=3$ situation variants, classical epistemic policy)",
    fontsize=10.5, fontweight="bold", pad=6, loc="left",
)

# ── Panel 1: stacked bar charts ───────────────────────────────────────────────

gs_bars = gs[1].subgridspec(1, 3, wspace=0.35)

mode_axes = [fig.add_subplot(gs_bars[0, k]) for k in range(3)]

model_labels = [label for label, _ in MODELS]
x = np.arange(len(model_labels))
bar_width = 0.55

for ax, (mode_key, mode_latex, mode_title) in zip(mode_axes, MODES):
    dist_key = f"{mode_key}_bilateral_distribution"

    # Build matrix: rows = models, cols = TV categories
    present = set()
    for label, _ in MODELS:
        present |= set(data[label][dist_key].keys())
    tv_cols = [tv for tv in TV_ORDER if tv in present]

    bottom = np.zeros(len(model_labels))
    patches = []
    for tv in tv_cols:
        counts = np.array([
            data[label][dist_key].get(tv, 0) for label, _ in MODELS
        ], dtype=float)
        pcts = counts / N * 100
        bars = ax.bar(x, pcts, bar_width, bottom=bottom,
                      color=TV_COLORS[tv], edgecolor="white", linewidth=0.5)
        bottom += pcts
        patches.append(mpatches.Patch(color=TV_COLORS[tv], label=TV_LABELS[tv]))

    ax.set_title(mode_title + "\n" + mode_latex, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of assertions" if ax == mode_axes[0] else "")
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, linewidth=0.7)

    # Annotate F1 above each bar stack
    for i, label in enumerate(model_labels):
        f1 = data[label][f"{mode_key}_f1_macro"]
        ax.text(i, bottom[i] + 1.5, f"F1={f1:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333")

# Shared legend below the bar panels
all_patches = []
for tv in TV_ORDER:
    if any(tv in data[label][f"{mode_key}_bilateral_distribution"]
           for label, _ in MODELS
           for mode_key, _, _ in MODES):
        all_patches.append(
            mpatches.Patch(color=TV_COLORS[tv], label=TV_LABELS[tv])
        )

fig.legend(
    handles=all_patches,
    loc="lower center",
    ncol=len(all_patches),
    fontsize=9,
    frameon=True,
    title="Bilateral truth value",
    title_fontsize=9,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "BBL Pilot: Bilateral Truth Value Distributions across Evaluation Modes",
    fontsize=12, fontweight="bold", y=0.97,
)

out = os.path.join(os.path.dirname(__file__), "results", "pilot_visualization.pdf")
out_png = out.replace(".pdf", ".png")
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"Saved: {out}")
print(f"Saved: {out_png}")
