"""SV branch validation progression — v1 → v2 → v3 (projected) → published refs."""
import matplotlib.pyplot as plt
import numpy as np

# (label, EER %, color, hatch)
runs = [
    ("v1\n(distill only,\n500 utts)",            28.67, "#B91C1C", ""),
    ("v2\n(distill + AAM,\n24K utts)",           10.64, "#B45309", ""),
    ("v3 target\n(ECAPA + tuned λ,\n~150K utts)", 4.50, "#65A30D", "//"),
    ("ref: x-vector\n(~4M params)",               4.00, "#475569", ""),
    ("ref: ECAPA-TDNN\n(6M params)",              1.00, "#1428A0", ""),
]

xs = list(range(len(runs)))
labels = [r[0] for r in runs]
eers   = [r[1] for r in runs]
colors = [r[2] for r in runs]
hatches = [r[3] for r in runs]

fig, ax = plt.subplots(figsize=(10, 4.6))
bars = ax.bar(xs, eers, color=colors, edgecolor="white", linewidth=1.5)
for bar, h in zip(bars, hatches):
    if h:
        bar.set_hatch(h)

# Annotate values
for i, v in enumerate(eers):
    ax.annotate(f"{v:.2f}%", (i, v), xytext=(0, 5),
                textcoords="offset points", ha="center",
                fontsize=10, fontweight="bold")

# Improvement arrows v1 -> v2 -> v3
ax.annotate("", xy=(0.6, 16.0), xytext=(0.0, 25.0),
            arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2))
ax.text(0.3, 21.0, "+18.0 pp\n(AAM)", fontsize=9, color="#16A34A",
        ha="center", fontweight="bold")

ax.annotate("", xy=(2.0, 5.5), xytext=(1.4, 10.0),
            arrowprops=dict(arrowstyle="->", color="#16A34A", lw=2,
                            linestyle="dashed"))
ax.text(1.7, 8.0, "+6 pp\n(projected)", fontsize=9, color="#16A34A",
        ha="center", style="italic")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("VoxCeleb1-O EER (%)", fontsize=11)
ax.set_title("SV branch validation — Phase-1 progression",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 32)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Hatched legend stripe = projected
ax.text(0.99, 0.97, "solid = measured     hatched = projected (v3)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, color="#475569", style="italic")

plt.tight_layout()
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/8_sv_progression.png",
            dpi=200, bbox_inches="tight")
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/8_sv_progression.svg",
            bbox_inches="tight")
print("wrote 8_sv_progression.{png,svg}")
