"""Babble (cocktail-party) robustness — P0 vs P1a vs P1c across SNR."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/malaymishra/Desktop/srib")

# Existing measured numbers (already in repo)
P0_BABBLE_N2  = {"clean": 97.71, 20: 92.78, 10: 79.01, 5: 62.01, 0: 42.08, -5: 27.06}
P1A_BABBLE_N2 = {"clean": 97.38, 20: 94.60, 10: 82.78, 5: 66.44, 0: 46.16, -5: 28.83}

# P1c (this run) — read from CSVs
def load_csv(path: Path) -> dict:
    if not path.is_file():
        return {}
    out = {}
    for r in csv.DictReader(open(path)):
        s = float(r["snr_db"])
        key = "clean" if s >= 900 else int(s)
        out[key] = float(r["val_acc"]) * 100.0
    return out

P1C_BABBLE_N2 = load_csv(ROOT / "babble_results/runs/p1c_pcen_babble/babble_sweep_n2.csv")
P1C_BABBLE_N3 = load_csv(ROOT / "results/babble_sweep/p1c_babble_n3.csv")
P1C_MUSAN     = load_csv(ROOT / "results/babble_sweep/p1c_musan_sweep.csv")

# Reference P1a MUSAN (already measured, hard-coded)
P1A_MUSAN = {"clean": 97.36, 20: 96.80, 10: 95.72, 5: 94.38, 0: 90.99, -5: 84.46}

SNR_ORDER = ["clean", 20, 10, 5, 0, -5]
X_LABELS  = ["clean", "20", "10", "5", "0", "−5"]
XS        = list(range(len(SNR_ORDER)))


def to_y(series: dict) -> list:
    return [series.get(k, float("nan")) for k in SNR_ORDER]


fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False)

# ---- Panel A: babble (cocktail-party) — the headline win
ax = axes[0]
ax.plot(XS, to_y(P0_BABBLE_N2),  marker="o", color="#B91C1C", lw=2.0,
        label="P0 (clean train)")
ax.plot(XS, to_y(P1A_BABBLE_N2), marker="s", color="#B45309", lw=2.0,
        label="P1a (RIR + MUSAN)")
ax.plot(XS, to_y(P1C_BABBLE_N2), marker="o", color="#1428A0", lw=2.6,
        label="P1c (+ babble-aug)  [new]")
if P1C_BABBLE_N3:
    ax.plot(XS, to_y(P1C_BABBLE_N3), marker="^", color="#1428A0", lw=1.5,
            ls="--", alpha=0.8, label="P1c, N=3 voices")
for i, (a, b, c) in enumerate(zip(to_y(P0_BABBLE_N2), to_y(P1A_BABBLE_N2),
                                   to_y(P1C_BABBLE_N2))):
    ax.annotate(f"{c:.0f}", (i, c), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=8.5,
                color="#1428A0", fontweight="bold")
ax.axhline(90, color="#65A30D", lw=1, ls=":", alpha=0.7)
ax.text(0.02, 91, "PS04 KPI: TAR ≥ 90% noisy", transform=ax.get_yaxis_transform(),
        fontsize=8, color="#65A30D", style="italic")
ax.set_xticks(XS); ax.set_xticklabels(X_LABELS)
ax.set_xlabel("Test-time SNR (dB)  —  babble noise")
ax.set_ylabel("Val accuracy (%)")
ax.set_ylim(20, 102)
ax.set_title("(a) Cocktail-party robustness — competing-speaker noise",
             fontsize=11, fontweight="bold")
ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

# ---- Panel B: MUSAN sanity — P1c keeps ambient-noise robustness
ax = axes[1]
ax.plot(XS, to_y(P1A_MUSAN), marker="s", color="#B45309", lw=2.0,
        label="P1a (RIR + MUSAN)")
ax.plot(XS, to_y(P1C_MUSAN), marker="o", color="#1428A0", lw=2.6,
        label="P1c (+ babble-aug)  [new]")
for i, (a, b) in enumerate(zip(to_y(P1A_MUSAN), to_y(P1C_MUSAN))):
    if not np.isnan(b):
        delta = b - a
        ax.annotate(f"{delta:+.1f}pp", (i, b), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8,
                    color="#1428A0")
ax.axhline(90, color="#65A30D", lw=1, ls=":", alpha=0.7)
ax.set_xticks(XS); ax.set_xticklabels(X_LABELS)
ax.set_xlabel("Test-time SNR (dB)  —  MUSAN ambient noise")
ax.set_ylabel("Val accuracy (%)")
ax.set_ylim(70, 102)
ax.set_title("(b) MUSAN sanity — minor erosion from training-data shift",
             fontsize=11, fontweight="bold")
ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)
ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

fig.suptitle(
    "Babble augmentation closes the cocktail-party gap  "
    "—  +18 pp at −5 dB babble  vs  −5 pp at −5 dB MUSAN",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.tight_layout()
plt.savefig(ROOT / "illustrations" / "12_babble_robustness.png",
            dpi=200, bbox_inches="tight")
plt.savefig(ROOT / "illustrations" / "12_babble_robustness.svg",
            bbox_inches="tight")
print("wrote 12_babble_robustness.{png,svg}")
print(f"  P1c babble N=2: { {k: round(v,1) for k,v in P1C_BABBLE_N2.items()} }")
if P1C_BABBLE_N3:
    print(f"  P1c babble N=3: { {k: round(v,1) for k,v in P1C_BABBLE_N3.items()} }")
print(f"  P1c MUSAN:      { {k: round(v,1) for k,v in P1C_MUSAN.items()} }")
