"""Generate 3090 prefill-heavy power-cap efficiency chart from @noonghunna's air-cooled rig.

Source data: 2026-05-07 sweep, 1× 3090 air-cooled (GPU 0).
Engine: mainline llama.cpp + Qwen3.6-27B Q3_K_XL.
Methodology: power-cap-sweep --load-mode prefill-heavy with adaptive prompt
calibration (probe at highest cap, size prompt for ~10s prefill at high cap,
use across sweep). Total wall: ~6m for 21 caps.

Companion to power-cap-3090-qwen36.png (decode-single curve from the same rig).
The two charts together tell the cross-workload story: same 3090 has different
sweet-spot at 290W for decode vs 250W for prefill.

Sampling fields: SM clock, memory clock, power-throttle %, P-state per cap
(median of in-load samples where util>50%). The boost-clock plateau visible at
caps 330-370W is now directly evidenced by SM clock locked at 1605-1620 MHz
across all five caps with identical 327W draw and 1050 prefill TPS — power is
binding (throttle=100%) but firmware refuses to push above ~1620 MHz until the
cap reaches 380W.
"""
import matplotlib.pyplot as plt

# (cap_W, prefill_TPS, actual_W, sm_clk_MHz, eff_TPS_per_W) — 21-cap clean sweep
data = [
    (190, 542.04, 189.73, 735, 2.857),
    (200, 606.05, 199.73, 825, 3.034),
    (210, 666.44, 209.69, 930, 3.178),
    (220, 732.61, 219.62, 1050, 3.336),
    (230, 796.31, 229.57, 1170, 3.469),
    (240, 855.79, 239.63, 1245, 3.571),
    (250, 906.79, 249.57, 1350, 3.633),  # ⭐ sweet spot
    (260, 934.60, 259.22, 1395, 3.605),
    (270, 959.59, 269.33, 1440, 3.563),
    (280, 982.50, 279.31, 1485, 3.518),
    (290, 1000.34, 289.54, 1515, 3.455),
    (300, 1015.32, 299.45, 1545, 3.391),
    (310, 1028.58, 309.31, 1560, 3.325),
    (320, 1041.55, 319.38, 1605, 3.261),
    (330, 1050.31, 327.08, 1605, 3.211),  # ←┐
    (340, 1051.35, 326.86, 1605, 3.217),  #   │ boost-clock plateau
    (350, 1050.13, 327.07, 1620, 3.211),  #   │ SM 1605/1620 MHz lock,
    (360, 1049.84, 326.91, 1620, 3.211),  #   │ 327W draw, 1050 TPS
    (370, 1051.07, 327.33, 1620, 3.211),  # ←┘ stock TDP
    (380, 1080.69, 354.99, 1665, 3.044),  # plateau ends, SM jumps to 1665
    (390, 1104.81, 381.19, 1710, 2.898),  # max — 381W draw, SM 1710
]

caps = [d[0] for d in data]
tps = [d[1] for d in data]
draw = [d[2] for d in data]
sm_clk = [d[3] for d in data]
eff = [d[4] for d in data]

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

# Left axis: prefill TPS
color_tps = "#7b3fa0"
ax1.plot(caps, tps, "o-", color=color_tps, linewidth=2.2, markersize=6,
         label="Prefill TPS (compute-bound)", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Prefill TPS (~11K-token prompt + max_tokens=10)", fontsize=13)
ax1.set_xlim(185, 395)
ax1.set_ylim(500, 1130)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (prefill TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: prefill TPS/W", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(2.7, 3.7)

# Sweet spot annotation: 250W
ax1.axvline(250, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 250W cap\n3.633 TPS/W (best efficiency)\n906.8 prefill TPS\nSM 1350 MHz, 68% of stock TDP",
    xy=(250, 906.79),
    xytext=(265, 660),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Boost-clock plateau region (330-370W → SM 1605/1620, 327W draw, 1050 TPS)
ax1.axvspan(325, 375, alpha=0.10, color="orange", zorder=0)
ax1.text(350, 525, "boost-clock plateau\n(caps 330-370W → SM locked at 1605-1620 MHz,\n327W draw, 1050 prefill TPS)",
         fontsize=9.5, ha="center", color="#aa5500", fontstyle="italic")

# Stock TDP marker
ax1.axvline(370, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n370W (GPU 0)",
    xy=(370, 1090),
    xytext=(372, 1095),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Plateau-escape annotation at 380W
ax1.annotate(
    "plateau escape:\nSM jumps 1620→1665 MHz",
    xy=(380, 1080.69),
    xytext=(330, 1115),
    fontsize=9,
    color="#aa5500",
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#aa5500", lw=0.9, alpha=0.7),
    zorder=4,
)

# Compare with decode sweet spot
ax1.text(295, 540, "(compare: decode-single sweet spot at 290W on same rig)",
         fontsize=9, ha="center", color="#666", fontstyle="italic")

# Title
ax1.set_title(
    "RTX 3090 + Qwen3.6-27B + llama.cpp — prefill-heavy power-cap curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090 air-cooled, mainline llama.cpp + Q3_K_XL GGUF, adaptive prompt sizing "
    "(11K tokens calibrated at 390W cap)  |  data: @noonghunna",
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

out = "/tmp/power_cap_sweep_3090_prefill.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
