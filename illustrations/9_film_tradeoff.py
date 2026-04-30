"""FiLM rejection trade-off — three operating points across recipe variants."""
import matplotlib.pyplot as plt
import numpy as np

# (label, TAR%, FAR%, REJ%, collapse_ratio, color)
runs = [
    ("v2 enrolled\n(self-cond train\n+ enrolled eval)",       62.50, 15.00, 85.00, 0.336, "#B45309"),
    ("v2 own-audio diag.\n(matched train/eval)\n[mechanism]", 94.64, 14.29, 85.71, 0.199, "#1428A0"),
    ("v3 paired enroll.\n(closes train/eval gap)",            92.14, 48.21, 51.07, 0.547, "#B91C1C"),
]

labels = [r[0] for r in runs]
tar    = [r[1] for r in runs]
far    = [r[2] for r in runs]
rej    = [r[3] for r in runs]
collapse = [r[4] for r in runs]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                               gridspec_kw={"width_ratios": [1.4, 1]})

# ----- Left: grouped bar chart (TAR / FAR / REJ for each variant)
xs = np.arange(len(runs))
W = 0.27
b1 = ax1.bar(xs - W, tar, W, color="#1E88E5", edgecolor="white", label="Genuine TAR")
b2 = ax1.bar(xs,     far, W, color="#B91C1C", edgecolor="white", label="Imposter FAR")
b3 = ax1.bar(xs + W, rej, W, color="#65A30D", edgecolor="white", label="Imposter REJ")
for bars in (b1, b2, b3):
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{bar.get_height():.0f}", ha="center", fontsize=8.5,
                 fontweight="bold")
ax1.set_xticks(xs); ax1.set_xticklabels(labels, fontsize=8.5)
ax1.set_ylabel("rate (%)"); ax1.set_ylim(0, 105)
ax1.set_title("FiLM operating points across recipe variants",
              fontsize=11, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
ax1.legend(loc="upper left", fontsize=9, framealpha=0.95)

# ----- Right: collapse-ratio interpretation
xs2 = np.arange(len(runs))
colors = [r[5] for r in runs]
b = ax2.bar(xs2, collapse, color=colors, edgecolor="white")
for bar, v in zip(b, collapse):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02,
             f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(xs2); ax2.set_xticklabels(["v2\nenroll", "v2\nown-audio", "v3\npaired"],
                                          fontsize=9)
ax2.set_ylabel("Collapse ratio  P(true_kw)_imp / P(true_kw)_gen")
ax2.set_ylim(0, 0.7)
ax2.set_title("Probability collapse — lower = stronger FiLM rejection",
              fontsize=11, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
ax2.axhline(0.20, color="#475569", linestyle=":", linewidth=1)
ax2.text(0.02, 0.21, "v2-diagnostic baseline (mechanism limit)",
         transform=ax2.get_yaxis_transform(), fontsize=8, style="italic",
         color="#475569")

plt.tight_layout()
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/9_film_tradeoff.png",
            dpi=200, bbox_inches="tight")
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/9_film_tradeoff.svg",
            bbox_inches="tight")
print("wrote 9_film_tradeoff.{png,svg}")
