"""Generate 5090 freq-cap (clock-locking) efficiency chart from @apnar's data.

Source data: 2026-05-07 sweep, 1× RTX 5090 air-cooled.
Engine: vLLM `gemma-mtp` compose + Gemma-4-31B-AutoRound + MTP K=3.
Methodology: instead of setting power caps via `nvidia-smi -pl`, lock GPU
SM clock + memory clock pairs via `nvidia-smi -lgc` and `-lmc`. This is a
workaround for the 5090's 400W power-cap floor (firmware refuses caps below
400W on this card). Clock-locking lets you reach operating points the
power-cap-sweep methodology can't access.

Why this matters:
  Power-cap sweet spot: 400W cap → 571 narr / 701 code TPS, 1.43 TPS/W
  Freq-cap sweet spot:  7001/1635 → 428 narr / 542 code TPS, 211W draw, 2.025 TPS/W
  → 1.42× more efficient at 47% less power, for 25% lower TPS

The clock-lock methodology surfaces a Pareto improvement over the power-cap
methodology on this card: 14001/2122 produces MORE TPS (602 vs 571) at LESS
power (314W vs ~400W), strictly dominating the power-cap sweet spot.

Data source: https://github.com/noonghunna/club-3090/discussions/86#discussioncomment-16845745
"""
import matplotlib.pyplot as plt

# (mem_MHz, gpu_MHz, narr_TPS, code_TPS, actual_W, eff_TPS_per_W)
data = [
    # 405 MHz mem (lowest) — bandwidth-starved at every GPU clock
    (405, 180,  27.65,  34.07,  47.69, 0.580),
    (405, 300,  28.69,  36.79,  53.02, 0.541),
    (405, 412,  29.49,  38.06,  55.59, 0.530),
    (405, 532,  29.65,  37.41,  51.71, 0.573),
    (405, 652,  29.25,  35.19,  57.22, 0.511),
    (405, 765,  52.96,  65.98,  63.18, 0.838),  # large jump — kernel threshold?
    (405, 885,  53.20,  68.08,  62.43, 0.852),

    # 810 MHz mem (low) — bandwidth ceiling clear above 667 GPU MHz
    (810, 180,  46.44,  58.78,  51.46, 0.902),
    (810, 667,  54.64,  65.50,  58.96, 0.927),
    (810, 1147, 55.25,  68.12,  68.09, 0.811),
    (810, 1635, 54.80,  67.99,  78.33, 0.700),
    (810, 2122, 56.30,  66.73,  87.59, 0.643),
    (810, 2602, 54.93,  67.84, 106.68, 0.515),
    (810, 3090, 54.93,  67.55, 142.80, 0.385),

    # 7001 MHz mem (mid) — efficiency knee at GPU 1635
    (7001, 180,   67.01,  82.64,  65.58, 1.022),
    (7001, 667,  212.69, 265.54, 117.47, 1.811),
    (7001, 1147, 327.81, 408.22, 162.04, 2.023),
    (7001, 1635, 428.19, 541.85, 211.47, 2.025),  # ⭐ peak efficiency
    (7001, 2122, 494.95, 605.76, 258.53, 1.914),
    (7001, 2602, 511.01, 626.28, 313.44, 1.630),
    (7001, 3090, 518.82, 657.63, 409.95, 1.266),

    # 14001 MHz mem (max) — TPS scales through tested GPU range
    (14001, 180,   66.32,  82.94,  80.31, 0.826),
    (14001, 667,  223.09, 274.49, 132.66, 1.682),
    (14001, 1147, 348.14, 429.73, 186.53, 1.866),
    (14001, 1635, 457.77, 562.38, 242.71, 1.886),
    (14001, 2122, 602.20, 737.58, 313.63, 1.920),  # ⭐ Pareto point vs 400W cap
    (14001, 2602, 716.51, 878.75, 417.07, 1.718),
    (14001, 3090, 805.21, 965.95, 558.95, 1.441),
]

# Group by mem clock for plotting
mem_tiers = sorted(set(d[0] for d in data))

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9.5), dpi=150,
                                gridspec_kw={"height_ratios": [3, 2], "hspace": 0.35})

mem_colors = {
    405:   "#9e9e9e",
    810:   "#fdae61",
    7001:  "#7b3fa0",
    14001: "#1f77b4",
}
mem_labels = {
    405:   "405 MHz mem (lowest)",
    810:   "810 MHz mem",
    7001:  "7001 MHz mem (50% of max)",
    14001: "14001 MHz mem (stock max)",
}

# Top panel: Narrative TPS vs GPU clock, one line per mem tier
for mem in mem_tiers:
    rows = sorted([d for d in data if d[0] == mem], key=lambda d: d[1])
    gpus = [d[1] for d in rows]
    narrs = [d[2] for d in rows]
    ax1.plot(gpus, narrs, "o-", color=mem_colors[mem],
             linewidth=2.0, markersize=6, label=mem_labels[mem], zorder=3)

ax1.set_xlabel("GPU SM clock (MHz, locked via -lgc)", fontsize=12)
ax1.set_ylabel("Narrative TPS (decode-concurrent N=6)", fontsize=12)
ax1.set_xlim(0, 3200)
ax1.set_ylim(0, 850)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=10)
ax1.legend(loc="upper left", fontsize=10.5, framealpha=0.95, edgecolor="#ccc",
           title="Memory clock", title_fontsize=10)
ax1.set_title("RTX 5090 + Gemma-4-31B + vLLM-MTP — freq-cap (clock-lock) sweep", pad=12)

# Sweet spot annotation: 7001 / 1635
ax1.axvline(1635, color="goldenrod", linestyle=":", alpha=0.4, linewidth=1.2)
ax1.scatter([1635], [428.19], s=180, color="goldenrod", marker="*",
            edgecolors="#aa5500", linewidths=1.5, zorder=5)
ax1.annotate(
    "★ peak efficiency\n7001/1635 → 2.025 TPS/W\n428 narr TPS, 211W draw\n(1.42× more efficient than\nthe 400W power-cap sweet spot)",
    xy=(1635, 428.19),
    xytext=(2350, 50),
    fontsize=9.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff3cd",
              edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.3),
    zorder=5,
)

# Pareto annotation: 14001 / 2122
ax1.scatter([2122], [602.20], s=180, color="#1f77b4", marker="*",
            edgecolors="#0a3d6e", linewidths=1.5, zorder=5)
ax1.annotate(
    "★ Pareto point\n14001/2122 → 1.92 TPS/W\n602 narr TPS, 314W draw\n(strictly better than\n400W cap:\n+5% TPS, -22% W)",
    xy=(2122, 602.20),
    xytext=(2350, 320),
    fontsize=9.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#dbeafe",
              edgecolor="#1f77b4", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.3),
    zorder=5,
)

# Bottom panel: efficiency
for mem in mem_tiers:
    rows = sorted([d for d in data if d[0] == mem], key=lambda d: d[1])
    gpus = [d[1] for d in rows]
    effs = [d[5] for d in rows]
    ax2.plot(gpus, effs, "^--", color=mem_colors[mem],
             linewidth=1.6, markersize=5, alpha=0.9, zorder=3)

ax2.set_xlabel("GPU SM clock (MHz, locked via -lgc)", fontsize=12)
ax2.set_ylabel("Efficiency: narrative TPS/W", fontsize=12)
ax2.set_xlim(0, 3200)
ax2.set_ylim(0, 2.2)
ax2.grid(True, alpha=0.3, zorder=0)
ax2.tick_params(axis="both", labelsize=10)
ax2.axhline(1.43, color="#888", linestyle=":", alpha=0.6, linewidth=1.2)
ax2.text(2700, 1.48, "400W power-cap\nsweet spot: 1.43 TPS/W",
         fontsize=9, color="#555", fontstyle="italic", ha="center")
ax2.scatter([1635], [2.025], s=140, color="goldenrod", marker="*",
            edgecolors="#aa5500", linewidths=1.2, zorder=5)
ax2.scatter([2122], [1.920], s=140, color="#1f77b4", marker="*",
            edgecolors="#0a3d6e", linewidths=1.2, zorder=5)
ax2.set_title("Efficiency (TPS/W) — clock-lock breaks past the 400W power-cap floor",
              fontsize=11, pad=8)

# Subtitle / data attribution
fig.text(
    0.5, 0.94,
    "1× RTX 5090 air-cooled, vLLM `gemma-mtp` + Gemma-4-31B-AutoRound + MTP K=3, "
    "decode-concurrent N=6, bench-runs=3  |  data: @apnar",
    ha="center", fontsize=10, color="#666",
    style="italic",
)

# Footer
fig.text(
    0.99, 0.005,
    "github.com/noonghunna/club-3090",
    ha="right", fontsize=9, color="#888", style="italic",
)

plt.tight_layout(rect=(0, 0.01, 1, 0.93))

out = "/tmp/freq_cap_5090_gemma4.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
