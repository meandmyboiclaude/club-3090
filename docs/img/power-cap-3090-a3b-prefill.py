"""Generate 3090 A3B MoE prefill-heavy power-cap chart.

Source data: 2026-05-08 sweep, 1× 3090 air-cooled (GPU 0).
Engine: mainline llama.cpp + Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf at -c 65536.
Methodology: power-cap-sweep --load-mode prefill-heavy with adaptive prompt
calibration (probe at 390W, ~31K-token prompt sized for 10s prefill at high
cap). Calibration probe: 10042 tok in 3.216s = 3122.51 TPS at 390W.

Companion to power-cap-3090-qwen36-prefill.png (dense Qwen3.6-27B prefill
on the same rig). Headline: prefill sweet spot is 250W on BOTH dense and
A3B MoE — same hardware, same workload class, same firmware optimum.
This contrasts with decode-single, where MoE shifts the sweet spot from
290W (dense) to 210W (MoE). Prefill is more compute-bound, so the knee
location is determined by hardware compute regime rather than model
architecture.

Boost-clock plateau detected at 340-370W: SM 1680-1710 MHz, 334W draw,
2802 TPS (auto-detected by power-cap-sweep.sh).
"""
import matplotlib.pyplot as plt

# (cap_W, prefill_TPS, actual_W, sm_clk_MHz, eff_TPS_per_W) — 21-cap clean sweep
data = [
    (190, 1542.94, 189.74, 720, 8.132),
    (200, 1690.30, 199.69, 855, 8.465),
    (210, 1860.96, 209.80, 945, 8.870),
    (220, 2039.30, 219.69, 1095, 9.283),
    (230, 2210.92, 229.54, 1215, 9.632),
    (240, 2356.70, 239.73, 1320, 9.831),
    (250, 2461.22, 249.49, 1380, 9.865),  # ⭐ sweet spot
    (260, 2523.67, 259.46, 1440, 9.727),
    (270, 2558.30, 269.56, 1485, 9.491),
    (280, 2611.96, 279.34, 1530, 9.350),
    (290, 2657.45, 288.95, 1560, 9.197),
    (300, 2699.15, 299.10, 1605, 9.024),
    (310, 2726.33, 308.69, 1635, 8.832),
    (320, 2763.36, 318.62, 1650, 8.673),
    (330, 2786.61, 328.53, 1665, 8.482),
    (340, 2802.41, 334.05, 1680, 8.389),  # ←┐
    (350, 2801.40, 334.89, 1710, 8.365),  #   │ boost-clock plateau (auto-detected)
    (360, 2797.63, 333.89, 1695, 8.379),  #   │ SM 1680-1710 MHz lock,
    (370, 2794.36, 334.15, 1695, 8.363),  # ←┘ 334W draw, 2802 TPS
    (380, 2866.93, 361.98, 1740, 7.920),
    (390, 2900.05, 387.58, 1785, 7.482),
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
         label="Prefill TPS (A3B MoE, compute-bound)", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Prefill TPS (~31K-token prompt + max_tokens=10)", fontsize=13)
ax1.set_xlim(185, 395)
ax1.set_ylim(1400, 3000)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (prefill TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: prefill TPS/W", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(7.3, 10.0)

# Sweet spot annotation: 250W
ax1.axvline(250, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 250W cap\n9.865 TPS/W (best efficiency)\n2461 prefill TPS\nSM 1380 MHz, 68% of stock TDP\n→ same as dense Qwen3.6-27B prefill sweet spot",
    xy=(250, 2461.22),
    xytext=(265, 1700),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Boost-clock plateau region (340-370W → SM 1680-1710, 334W draw, 2802 TPS)
ax1.axvspan(335, 375, alpha=0.10, color="orange", zorder=0)
ax1.text(355, 1430, "boost-clock plateau\n(caps 340-370W → SM locked at 1680-1710 MHz,\n334W draw, 2802 prefill TPS)",
         fontsize=9.5, ha="center", color="#aa5500", fontstyle="italic")

# Stock TDP marker
ax1.axvline(370, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n370W (GPU 0)",
    xy=(370, 2900),
    xytext=(372, 2920),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Plateau-escape annotation at 380W
ax1.annotate(
    "plateau escape:\nSM jumps 1710→1740 MHz",
    xy=(380, 2866.93),
    xytext=(330, 2950),
    fontsize=9,
    color="#aa5500",
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#aa5500", lw=0.9, alpha=0.7),
    zorder=4,
)

# Title
ax1.set_title(
    "RTX 3090 + Qwen3.6-35B-A3B (MoE) + llama.cpp — prefill-heavy power-cap curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090 air-cooled, mainline llama.cpp + A3B Q4_K_XL GGUF (-c 65536), adaptive prompt sizing "
    "(31K tokens calibrated at 390W cap)  |  data: @noonghunna",
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

out = "/tmp/power_cap_sweep_3090_a3b_prefill.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
