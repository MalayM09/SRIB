"""Render parameter-budget horizontal bar chart for blueprint slide 4."""
import matplotlib.pyplot as plt

components = [
    ("BC-ResNet-8 trunk (shared, validated)", 323, "#4338ca"),
    ("KWS branch (2 blocks)",                 180, "#65a30d"),
    ("SV branch (2 blocks)",                  180, "#65a30d"),
    ("MQMHA pooling",                          30, "#10b981"),
    ("SV head + projection",                   55, "#6d28d9"),
    ("KWS head + FiLM generator",              25, "#b91c1c"),
    ("TRM_kws + TRM_sv",                       15, "#b45309"),
    ("ECAPA distill proj heads",               15, "#0891b2"),
    ("WavLM hint-layer proj",                  10, "#0891b2"),
    ("Learnable PCEN params",                   0.2, "#64748b"),
]

components = sorted(components, key=lambda x: x[1], reverse=True)
labels  = [c[0] for c in components]
params  = [c[1] for c in components]
colors  = [c[2] for c in components]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.barh(labels, params, color=colors, edgecolor="white")
ax.invert_yaxis()
ax.set_xlabel("Parameters (thousands)", fontsize=11)

total = sum(params)
ax.set_title(
    f"Student Parameter Budget — Total ≈ {total:.1f}K ({total/1000:.2f}M)   |   Cap: 3.00M",
    fontsize=12, fontweight="bold"
)

for bar, val in zip(bars, params):
    ax.text(val + 8, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}K", va="center", fontsize=10)

ax.set_xlim(0, max(params) * 1.18)
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# headroom annotation
headroom_pct = (1 - total/3000) * 100
ax.text(0.99, 0.02,
        f"Headroom vs. 3M cap: {headroom_pct:.0f}%  (for QAT scale/zero-point tensors)",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", color="#475569")

plt.tight_layout()
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/6_param_budget.png", dpi=200, bbox_inches="tight")
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/6_param_budget.svg", bbox_inches="tight")
print(f"Rendered param budget — total {total:.1f}K params")
