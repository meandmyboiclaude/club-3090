"""Generate 3090 power-cap efficiency chart from @noonghunna's air-cooled rig.

Source data: 2026-05-07 sweep, dual-3090 rig (GPU 0 used), air-cooled.
Engine: mainline llama.cpp (ghcr.io/ggml-org/llama.cpp:server-cuda) +
Qwen3.6-27B-UD-Q3_K_XL.gguf, single-stream decode-single.

Sweep methodology: time-bounded streaming bench (10s/direction at each cap).
Total wall: 8m12s for 21 caps from 190-390W in 10W increments. The time-bounded
approach (vs token-bounded) makes per-cap wall constant ~23s regardless of cap,
so total runtime scales linearly with cap count, not throttle severity.

Sampling fields: SM clock, memory clock, power-throttle %, P-state per cap
(median of in-load samples where util>50%). The boost-clock plateau visible at
caps 340-370W is now directly evidenced by SM clock locked at 1560 MHz across
all four caps, with identical 334W actual draw and 34.66 TPS — power is binding
at every cap (throttle=100%) but firmware refuses to push above 1560 MHz until
the cap reaches 380W.
"""
import matplotlib.pyplot as plt

# (cap_W, narr_TPS, code_TPS, actual_W, sm_clk_MHz, eff_TPS_per_W) — 21-cap clean sweep
data = [
    (190, 14.19, 13.78, 189.71, 540, 0.075),
    (200, 15.68, 15.58, 199.72, 585, 0.079),
    (210, 17.48, 17.38, 209.69, 660, 0.083),
    (220, 19.28, 19.48, 219.71, 750, 0.088),
    (230, 21.58, 21.37, 229.73, 825, 0.094),
    (240, 23.38, 23.17, 239.74, 915, 0.098),
    (250, 25.17, 25.07, 249.80, 975, 0.101),
    (260, 27.07, 26.97, 259.87, 1080, 0.104),
    (270, 28.97, 28.87, 269.86, 1185, 0.107),
    (280, 30.77, 30.67, 279.76, 1260, 0.110),
    (290, 32.26, 32.17, 289.55, 1380, 0.111),  # ⭐ sweet spot
    (300, 32.96, 32.87, 299.22, 1425, 0.110),
    (310, 33.57, 33.47, 309.57, 1470, 0.108),
    (320, 34.06, 33.96, 319.41, 1515, 0.107),
    (330, 34.57, 34.37, 329.42, 1545, 0.105),
    (340, 34.66, 34.66, 334.08, 1560, 0.104),  # ←┐
    (350, 34.67, 34.67, 334.09, 1560, 0.104),  #   │ boost-clock plateau
    (360, 34.66, 34.67, 333.93, 1560, 0.104),  #   │ SM clk locks at 1560 MHz
    (370, 34.66, 34.67, 334.00, 1560, 0.104),  # ←┘ stock TDP
    (380, 35.56, 35.46, 361.67, 1635, 0.098),  # plateau ends, SM jumps to 1635
    (390, 36.26, 36.06, 388.53, 1680, 0.093),  # max — SM 1680 MHz at 388W draw
]

caps = [d[0] for d in data]
narr = [d[1] for d in data]
code = [d[2] for d in data]
draw = [d[3] for d in data]
sm_clk = [d[4] for d in data]
eff = [d[5] for d in data]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax1 = plt.subplots(figsize=(11, 6.4), dpi=150)

# Left axis: TPS
color_narr = "#1f77b4"
color_code = "#2ca02c"
ax1.plot(caps, narr, "o-", color=color_narr, linewidth=2.2, markersize=6,
         label="Narrative TPS", zorder=3)
ax1.plot(caps, code, "s-", color=color_code, linewidth=2.2, markersize=6,
         label="Code TPS", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Wall TPS (single-stream, llama.cpp mainline)", fontsize=13)
ax1.set_xlim(185, 395)
ax1.set_ylim(11, 39)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: TPS/W efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (narr TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: TPS/W (narrative)", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(0.07, 0.118)

# Sweet spot annotation: 290W
ax1.axvline(290, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 290W cap\n0.111 TPS/W (best efficiency)\n32.3 narr / 32.2 code\nSM 1380 MHz, 78% of stock TDP",
    xy=(290, 32.26),
    xytext=(217, 27),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Boost-clock plateau region (340-370W → all SM 1560 MHz, 334W draw, 34.66 TPS)
ax1.axvspan(335, 375, alpha=0.10, color="orange", zorder=0)
ax1.text(355, 12.7, "boost-clock plateau\n(caps 340-370W → SM locked\nat 1560 MHz, 334W draw, 34.66 TPS)",
         fontsize=9.5, ha="center", color="#aa5500", fontstyle="italic")

# Stock TDP marker at 370W
ax1.axvline(370, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n370W (GPU 0)",
    xy=(370, 36.5),
    xytext=(372, 36.8),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Plateau-escape annotation at 380W
ax1.annotate(
    "plateau escape:\nSM jumps 1560→1635 MHz",
    xy=(380, 35.56),
    xytext=(345, 38),
    fontsize=9,
    color="#aa5500",
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#aa5500", lw=0.9, alpha=0.7),
    zorder=4,
)

# Title
ax1.set_title(
    "RTX 3090 + Qwen3.6-27B + llama.cpp — power-cap efficiency curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090 air-cooled (GPU 0 of dual-3090 rig), mainline llama.cpp + Q3_K_XL GGUF, "
    "time-bounded single-stream  |  data: @noonghunna",
    ha="center", fontsize=10, color="#666",
    style="italic",
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=11, framealpha=0.95,
           edgecolor="#ccc")

# Footer
fig.text(
    0.99, 0.01,
    "github.com/noonghunna/club-3090",
    ha="right", fontsize=9, color="#888", style="italic",
)

plt.tight_layout(rect=(0, 0.02, 1, 0.92))

out = "/tmp/power_cap_sweep_3090_qwen36.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
