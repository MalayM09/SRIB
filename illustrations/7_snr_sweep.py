"""Render the Phase-1 empirical-validation two-panel figure.

Panel (a) — training-regime effect: noise-aug training vs clean training,
averaged across PCEN + log-Mel front-ends. Shows the dominant robustness
lever: data diversity at training time.

Panel (b) — PCEN vs log-Mel ablation under noise-aug training, mean ±
halfspread across seeds {1337, 42}. Shows that PCEN beats log-Mel in
direction at every SNR and has visibly lower cross-seed variance.
"""
import matplotlib.pyplot as plt
import numpy as np

SNR_ORDER  = ["clean", 20, 10, 5, 0, -5]
X_LABELS   = ["clean", "20", "10", "5", "0", "-5"]
XS         = list(range(len(SNR_ORDER)))

# -------------------------------------------------- raw per-run SNR sweeps
# Noise-augmented training (RIR + MUSAN), two seeds
P1A_PCEN_S1337   = {"clean": 0.9736, 20: 0.9680, 10: 0.9572, 5: 0.9438, 0: 0.9099, -5: 0.8446}
P1A_LOGMEL_S1337 = {"clean": 0.9728, 20: 0.9652, 10: 0.9527, 5: 0.9404, 0: 0.9034, -5: 0.8329}
P1A_PCEN_S42     = {"clean": 0.9762, 20: 0.9706, 10: 0.9609, 5: 0.9442, 0: 0.9140, -5: 0.8472}
P1A_LOGMEL_S42   = {"clean": 0.9743, 20: 0.9700, 10: 0.9540, 5: 0.9410, 0: 0.9099, -5: 0.8440}

# Clean training (SpecAugment only, no RIR/MUSAN), one seed
P1B_PCEN   = {"clean": 0.9767, 20: 0.9635, 10: 0.9298, 5: 0.8902, 0: 0.8033, -5: 0.6546}
P1B_LOGMEL = {"clean": 0.9771, 20: 0.9598, 10: 0.9293, 5: 0.8896, 0: 0.8014, -5: 0.6544}


def pct(series):
    return np.array([series[k] * 100 for k in SNR_ORDER])


# -------------------------------------------------- aggregates
# Panel A: mean across front-ends AND seeds for noise-aug; mean across
# front-ends for clean (only one seed run).
noise_aug_all = np.stack([pct(P1A_PCEN_S1337), pct(P1A_LOGMEL_S1337),
                          pct(P1A_PCEN_S42),   pct(P1A_LOGMEL_S42)])
clean_all     = np.stack([pct(P1B_PCEN), pct(P1B_LOGMEL)])
noise_aug_mean = noise_aug_all.mean(axis=0)
clean_mean     = clean_all.mean(axis=0)

# Panel B: mean ± halfspread over 2 seeds, per front-end, under noise-aug
pcen_stack   = np.stack([pct(P1A_PCEN_S1337),   pct(P1A_PCEN_S42)])
logmel_stack = np.stack([pct(P1A_LOGMEL_S1337), pct(P1A_LOGMEL_S42)])
pcen_mean, pcen_hs     = pcen_stack.mean(axis=0),   (pcen_stack.max(axis=0)   - pcen_stack.min(axis=0)) / 2
logmel_mean, logmel_hs = logmel_stack.mean(axis=0), (logmel_stack.max(axis=0) - logmel_stack.min(axis=0)) / 2


# -------------------------------------------------- plot
PCEN_COLOR   = "#1E88E5"
LOGMEL_COLOR = "#B45309"
NOISE_COLOR  = "#1428A0"
CLEAN_COLOR  = "#B91C1C"

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=False)

# ---- Panel A: training regime effect
ax = axes[0]
ax.plot(XS, noise_aug_mean, marker="o", color=NOISE_COLOR, lw=2.4,
        label="Noise-aug training  (RIR + MUSAN)")
ax.plot(XS, clean_mean,     marker="s", color=CLEAN_COLOR, lw=2.4, ls="--",
        label="Clean training  (SpecAugment only)")
for i, (n, c) in enumerate(zip(noise_aug_mean, clean_mean)):
    ax.annotate(f"{n:.1f}", (i, n), xytext=(0, 7),  textcoords="offset points",
                ha="center", fontsize=8, color=NOISE_COLOR)
    ax.annotate(f"{c:.1f}", (i, c), xytext=(0, -13), textcoords="offset points",
                ha="center", fontsize=8, color=CLEAN_COLOR)
# highlight the -5 dB gap
gap_pp = noise_aug_mean[-1] - clean_mean[-1]
ax.annotate(f"+{gap_pp:.0f} pp",
            xy=(len(XS) - 1, (noise_aug_mean[-1] + clean_mean[-1]) / 2),
            xytext=(len(XS) - 2.2, (noise_aug_mean[-1] + clean_mean[-1]) / 2),
            fontsize=11, fontweight="bold", color="#000",
            arrowprops=dict(arrowstyle="-", color="#555", lw=1))
ax.set_xticks(XS); ax.set_xticklabels(X_LABELS)
ax.set_xlabel("Test-time SNR (dB)", fontsize=10)
ax.set_ylabel("Val accuracy (%)", fontsize=10)
ax.set_title("(a) Training regime dominates — data diversity is the main robustness lever",
             fontsize=10.5, fontweight="bold")
ax.set_ylim(60, 100)
ax.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

# ---- Panel B: PCEN vs log-Mel under noise-aug, across seeds
ax = axes[1]
ax.plot(XS, pcen_mean,   marker="o", color=PCEN_COLOR,   lw=2.4, label="PCEN")
ax.fill_between(XS, pcen_mean - pcen_hs, pcen_mean + pcen_hs,
                color=PCEN_COLOR, alpha=0.18)
ax.plot(XS, logmel_mean, marker="s", color=LOGMEL_COLOR, lw=2.4, label="log-Mel")
ax.fill_between(XS, logmel_mean - logmel_hs, logmel_mean + logmel_hs,
                color=LOGMEL_COLOR, alpha=0.18)
for i, (p, l) in enumerate(zip(pcen_mean, logmel_mean)):
    d_pp = p - l
    ax.annotate(f"+{d_pp:.2f}", (i, max(p, l)), xytext=(0, 9),
                textcoords="offset points", ha="center", fontsize=8, color="#111")
ax.set_xticks(XS); ax.set_xticklabels(X_LABELS)
ax.set_xlabel("Test-time SNR (dB)", fontsize=10)
ax.set_title("(b) PCEN vs log-Mel  (noise-aug, mean ± halfspread over 2 seeds)",
             fontsize=10.5, fontweight="bold")
ax.set_ylim(80, 100)
ax.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

fig.suptitle(
    "Phase-1 Empirical Validation  —  SC V2, 12 classes, BC-ResNet-8 @ 324 K params",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/7_snr_sweep.png",
            dpi=200, bbox_inches="tight")
plt.savefig("/Users/malaymishra/Desktop/srib/illustrations/7_snr_sweep.svg",
            bbox_inches="tight")
print(f"Panel A gap at -5 dB: +{gap_pp:.1f} pp  (noise-aug {noise_aug_mean[-1]:.1f}% vs clean {clean_mean[-1]:.1f}%)")
print(f"Panel B PCEN delta mean across SNRs: +{(pcen_mean - logmel_mean).mean():.2f} pp")
print("wrote 7_snr_sweep.{png,svg}")
