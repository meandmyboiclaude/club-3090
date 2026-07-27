"""Generate 3090 A3B MoE decode-single power-cap chart.

Source data: 2026-05-08 sweep, dual-3090 rig (GPU 0 used), air-cooled.
Engine: mainline llama.cpp (ghcr.io/ggml-org/llama.cpp:server-cuda) +
Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf (MoE, 3B active per token), single-stream.

Companion to power-cap-3090-qwen36.png (dense Qwen3.6-27B on the same rig).
The headline finding: same hardware, MoE workload shifts sweet spot
from 290W (dense) → 210W (MoE) — an 80W gap driven by the lower
compute-per-token of the MoE path (bandwidth becomes the binding
constraint earlier).

Notable difference from the dense decode chart: NO firmware boost-clock
plateau in the 340-370W range. SM clock climbs smoothly across that
range (1875 → 1890 → 1890 → 1905) instead of locking at a single value.
Plateau auto-detection correctly flagged dense Qwen but did NOT flag
A3B — confirming that firmware operating-point selection responds to
the model's compute profile, not just to the cap value.
"""
import matplotlib.pyplot as plt

# (cap_W, narr_TPS, code_TPS, actual_W, sm_clk_MHz, eff_TPS_per_W) — 21-cap clean sweep
data = [
    (190, 92.20, 91.60, 189.72, 900, 0.486),
    (200, 104.09, 104.08, 199.73, 1095, 0.521),
    (210, 114.59, 113.79, 209.71, 1290, 0.546),  # ⭐ sweet spot
    (220, 119.46, 119.07, 219.70, 1425, 0.544),
    (230, 122.57, 122.68, 229.69, 1530, 0.534),
    (240, 124.78, 124.95, 239.63, 1590, 0.521),
    (250, 127.16, 126.99, 249.55, 1635, 0.510),
    (260, 128.45, 128.56, 259.55, 1680, 0.495),
    (270, 130.26, 129.97, 269.37, 1710, 0.484),
    (280, 131.46, 131.14, 279.26, 1740, 0.471),
    (290, 132.23, 132.25, 289.27, 1755, 0.457),
    (300, 133.27, 133.23, 298.96, 1785, 0.446),
    (310, 134.07, 134.07, 309.19, 1815, 0.434),
    (320, 134.94, 135.06, 318.88, 1845, 0.423),
    (330, 135.54, 135.65, 328.96, 1860, 0.412),
    (340, 136.25, 135.84, 338.71, 1875, 0.402),
    (350, 136.55, 136.35, 348.73, 1890, 0.392),
    (360, 136.66, 136.65, 354.04, 1890, 0.386),
    (370, 136.84, 136.65, 354.23, 1905, 0.386),
    (380, 137.26, 137.46, 377.33, 1920, 0.364),
    (390, 137.73, 137.85, 385.80, 1935, 0.357),
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
color_narr = "#7b3fa0"
color_code = "#1f77b4"
ax1.plot(caps, narr, "o-", color=color_narr, linewidth=2.2, markersize=6,
         label="Narrative TPS (A3B MoE)", zorder=3)
ax1.plot(caps, code, "s-", color=color_code, linewidth=2.2, markersize=6,
         label="Code TPS (A3B MoE)", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Wall TPS (single-stream, llama.cpp mainline)", fontsize=13)
ax1.set_xlim(185, 395)
ax1.set_ylim(85, 145)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: TPS/W efficiency
ax2 = ax1.twinx()
color_eff = "#d62728"
ax2.plot(caps, eff, "^--", color=color_eff, linewidth=1.8, markersize=5,
         alpha=0.9, label="Efficiency (narr TPS/W)", zorder=2)
ax2.set_ylabel("Efficiency: TPS/W (narrative)", color=color_eff, fontsize=13)
ax2.tick_params(axis="y", labelcolor=color_eff, labelsize=11)
ax2.set_ylim(0.34, 0.58)

# Sweet spot annotation: 210W
ax1.axvline(210, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 210W cap\n0.546 TPS/W (best efficiency)\n114.6 narr / 113.8 code\nSM 1290 MHz, 57% of stock TDP\n→ 80W lower than dense Qwen3.6-27B",
    xy=(210, 114.59),
    xytext=(225, 88),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# No-plateau annotation
ax1.axvspan(335, 375, alpha=0.10, color="#9b59b6", zorder=0)
ax1.text(355, 142.5, "no boost-clock plateau\n(SM climbs 1875→1905,\nunlike dense Qwen at this range)",
         fontsize=9.5, ha="center", color="#5b3578", fontstyle="italic")

# Stock TDP marker at 370W
ax1.axvline(370, color="#888", linestyle="--", alpha=0.6, linewidth=1.2)
ax1.annotate(
    "stock TDP\n370W (GPU 0)",
    xy=(370, 137),
    xytext=(372, 137.5),
    fontsize=10,
    ha="left",
    color="#555",
    fontstyle="italic",
)

# Compare with dense Qwen sweet spot
ax1.text(295, 87, "(compare: dense Qwen3.6-27B sweet spot at 290W on same rig)",
         fontsize=9, ha="center", color="#666", fontstyle="italic")

# Title
ax1.set_title(
    "RTX 3090 + Qwen3.6-35B-A3B (MoE) + llama.cpp — power-cap efficiency curve",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 3090 air-cooled (GPU 0 of dual-3090 rig), mainline llama.cpp + A3B Q4_K_XL GGUF, "
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

out = "/tmp/power_cap_sweep_3090_a3b_decode.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
