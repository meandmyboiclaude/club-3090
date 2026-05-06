"""Generate a tweet-ready chart from @laurimyllari's RTX 4090 power-cap sweep on llama.cpp.

Source data: https://github.com/noonghunna/club-3090/discussions/62#discussioncomment-16854218
Setup: 1× RTX 4090 air-cooled, llama.cpp default, Qwen3.6-27B-UD-Q3_K_XL GGUF, sweep on club-3090@aa99173.

Two load modes are overlaid:
  - decode-single     (10s × 2 timed streams per cap)
  - decode-concurrent (concurrency=4, 4s/run × 3 runs × 2 batches per cap)

The two curves together show: (a) where the efficiency knee is (250-260W on either mode),
(b) the firmware boost-clock plateau locks at SM 2610 MHz / 393W draw from cap=400W onwards,
and (c) on this 27B model concurrency=4 sits below single-stream — i.e. the 4090 is in
under-load territory at c=4 on Q3_K_XL.
"""
import matplotlib.pyplot as plt

# decode-single: 38-cap sweep, 230-600W. Columns: cap, narr_tps, code_tps, actual_W, narr_tps_per_W
# Source: disc #62 16854218 (decode-single 14:42:23Z run)
decode_single = [
    (230, 41.47, 41.47, 229.72, 0.181),
    (240, 43.47, 43.77, 240.21, 0.181),
    (250, 46.37, 46.36, 249.84, 0.186),
    (260, 48.26, 48.16, 259.90, 0.186),  # sweet spot (tied with 250W)
    (270, 48.96, 48.96, 269.92, 0.181),
    (280, 49.36, 49.36, 279.72, 0.176),
    (290, 49.76, 49.86, 289.63, 0.172),
    (300, 50.16, 50.16, 299.69, 0.167),
    (310, 50.46, 50.56, 309.58, 0.163),
    (320, 50.76, 50.76, 319.34, 0.159),
    (330, 51.06, 51.06, 329.55, 0.155),
    (340, 51.26, 51.26, 339.47, 0.151),
    (350, 51.36, 51.36, 349.67, 0.147),
    (360, 51.56, 51.56, 359.38, 0.143),
    (370, 51.66, 51.66, 369.43, 0.140),
    (380, 51.86, 51.86, 378.50, 0.137),
    (390, 51.95, 51.96, 387.96, 0.134),
    (400, 51.96, 51.96, 392.17, 0.132),  # firmware plateau begins
    (410, 51.96, 51.96, 391.69, 0.133),
    (420, 51.96, 51.96, 392.14, 0.133),
    (430, 51.96, 51.96, 392.13, 0.133),
    (440, 51.95, 51.96, 393.26, 0.132),
    (450, 51.96, 51.96, 393.25, 0.132),
    (460, 51.86, 51.86, 393.58, 0.132),
    (470, 51.96, 51.96, 393.19, 0.132),
    (480, 51.85, 51.96, 394.23, 0.132),
    (490, 51.86, 51.96, 394.13, 0.132),
    (500, 51.86, 51.96, 393.15, 0.132),
    (510, 51.86, 51.86, 394.03, 0.132),
    (520, 51.86, 51.96, 393.96, 0.132),
    (530, 51.85, 51.96, 394.25, 0.132),
    (540, 51.86, 51.96, 394.76, 0.131),
    (550, 51.86, 51.96, 394.88, 0.131),
    (560, 51.86, 51.96, 394.58, 0.131),
    (570, 51.85, 51.96, 394.33, 0.131),
    (580, 51.86, 51.96, 394.32, 0.132),
    (590, 51.86, 51.96, 394.52, 0.131),
    (600, 51.85, 51.96, 394.87, 0.131),
]

# decode-concurrent N=4: 38-cap sweep, 230-600W.
# Source: disc #62 16854218 (decode-concurrent 14:58:09Z run, concurrency=4 auto-calibrated)
decode_concurrent = [
    (230, 36.96, 36.46, 229.58, 0.161),
    (240, 38.92, 38.69, 239.79, 0.162),
    (250, 41.14, 40.66, 249.65, 0.165),  # under-load knee (tied with 260W)
    (260, 42.13, 42.35, 259.50, 0.162),
    (270, 43.11, 43.34, 269.13, 0.160),
    (280, 43.60, 43.61, 278.92, 0.156),
    (290, 44.10, 43.85, 289.16, 0.153),
    (300, 44.58, 44.35, 299.48, 0.149),
    (310, 44.59, 44.83, 309.32, 0.144),
    (320, 44.83, 45.08, 319.19, 0.140),
    (330, 45.08, 45.07, 328.98, 0.137),
    (340, 45.34, 45.33, 339.07, 0.134),
    (350, 45.59, 45.58, 349.02, 0.131),
    (360, 45.82, 45.57, 358.57, 0.128),
    (370, 45.82, 45.81, 368.40, 0.124),
    (380, 45.82, 46.06, 378.29, 0.121),
    (390, 46.29, 46.07, 387.93, 0.119),
    (400, 46.08, 46.06, 393.64, 0.117),  # plateau begins
    (410, 46.31, 46.06, 394.30, 0.117),
    (420, 46.06, 46.07, 394.04, 0.117),
    (430, 46.07, 46.07, 395.27, 0.117),
    (440, 46.08, 46.07, 394.50, 0.117),
    (450, 46.08, 46.07, 394.64, 0.117),
    (460, 46.07, 46.04, 394.67, 0.117),
    (470, 46.04, 46.08, 394.48, 0.117),
    (480, 46.07, 46.08, 394.49, 0.117),
    (490, 46.08, 46.07, 394.24, 0.117),
    (500, 46.32, 46.07, 394.13, 0.118),
    (510, 46.07, 46.08, 394.65, 0.117),
    (520, 46.08, 46.07, 395.27, 0.117),
    (530, 46.08, 46.06, 394.79, 0.117),
    (540, 46.06, 46.07, 395.07, 0.117),
    (550, 46.07, 46.07, 395.58, 0.116),
    (560, 46.08, 46.07, 395.36, 0.117),
    (570, 46.07, 46.07, 395.51, 0.116),
    (580, 46.07, 46.07, 394.87, 0.117),
    (590, 46.08, 46.07, 395.44, 0.117),
    (600, 46.07, 46.06, 395.51, 0.116),
]

ds_caps = [d[0] for d in decode_single]
ds_narr = [d[1] for d in decode_single]
ds_eff = [d[4] for d in decode_single]

dc_caps = [d[0] for d in decode_concurrent]
dc_narr = [d[1] for d in decode_concurrent]
dc_eff = [d[4] for d in decode_concurrent]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

fig, ax1 = plt.subplots(figsize=(12, 6.5), dpi=150)

# Firmware boost-clock plateau shading (caps 400-600W)
ax1.axvspan(395, 605, color="#ffd6a5", alpha=0.35, zorder=0,
            label="Firmware boost-clock plateau (SM 2610 MHz / 393W draw)")

# Left axis: TPS (both modes)
color_ds = "#1f77b4"
color_dc = "#2ca02c"
ax1.plot(ds_caps, ds_narr, "o-", color=color_ds, linewidth=2.2, markersize=5,
         label="decode-single (narr TPS)", zorder=3)
ax1.plot(dc_caps, dc_narr, "s-", color=color_dc, linewidth=2.2, markersize=5,
         label="decode-concurrent N=4 (narr TPS)", zorder=3)
ax1.set_xlabel("Power cap (W)", fontsize=13)
ax1.set_ylabel("Narrative wall TPS", fontsize=13)
ax1.set_xlim(225, 605)
ax1.set_ylim(35, 54)
ax1.grid(True, alpha=0.3, zorder=0)
ax1.tick_params(axis="both", labelsize=11)

# Right axis: TPS/W efficiency (both modes)
ax2 = ax1.twinx()
color_eff_ds = "#d62728"
color_eff_dc = "#9467bd"
ax2.plot(ds_caps, ds_eff, "^--", color=color_eff_ds, linewidth=1.5, markersize=4,
         alpha=0.85, label="decode-single TPS/W", zorder=2)
ax2.plot(dc_caps, dc_eff, "v--", color=color_eff_dc, linewidth=1.5, markersize=4,
         alpha=0.85, label="decode-concurrent TPS/W", zorder=2)
ax2.set_ylabel("Efficiency: TPS/W (narrative)", fontsize=13, color="#555")
ax2.tick_params(axis="y", labelcolor="#555", labelsize=11)
ax2.set_ylim(0.10, 0.20)

# Sweet spot annotation: 260W on decode-single
ax1.axvline(260, color="goldenrod", linestyle=":", alpha=0.5, linewidth=1.5)
ax1.annotate(
    "★ 260W\n0.186 TPS/W\n(decode-single knee)\n42% below stock TDP",
    xy=(260, 48.26),
    xytext=(295, 50.5),
    fontsize=10.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="goldenrod", linewidth=1.2),
    arrowprops=dict(arrowstyle="->", color="goldenrod", lw=1.5),
    zorder=4,
)

# Plateau annotation
ax1.annotate(
    "Caps 400-600W functionally identical\n(SM 2610 MHz / 393W actual draw)",
    xy=(500, 51.96),
    xytext=(450, 47.5),
    fontsize=9.5,
    ha="center",
    color="#a85a00",
    fontstyle="italic",
    arrowprops=dict(arrowstyle="->", color="#cc8800", lw=1),
)

# Title
ax1.set_title(
    "RTX 4090 + Qwen3.6-27B + llama.cpp — power-cap efficiency curves",
    pad=14,
)

# Subtitle
fig.text(
    0.5, 0.92,
    "1× 4090 air-cooled, llama.cpp default, Q3_K_XL GGUF, 38-cap sweep at 10W  |  data: @laurimyllari (club-3090 disc #62)",
    ha="center", fontsize=10, color="#666",
    style="italic",
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc="lower right", fontsize=10, framealpha=0.95,
           edgecolor="#ccc", ncol=1)

# Footer
fig.text(
    0.99, 0.01,
    "github.com/noonghunna/club-3090",
    ha="right", fontsize=9, color="#888", style="italic",
)

plt.tight_layout(rect=(0, 0.02, 1, 0.92))

out = "docs/img/power-cap-4090-qwen36.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
