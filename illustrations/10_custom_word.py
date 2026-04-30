"""Custom-word few-shot detection — Phase-1 validation figure.

Loads results from results/custom_word_poc/ and renders a 3-panel slide-quality
plot: score distributions, per-held-out-word EER, ROC curve.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/Users/malaymishra/Desktop/srib")
SUM  = json.load(open(ROOT / "results" / "custom_word_poc" / "custom_word_summary.json"))
CSV_PATH = ROOT / "results" / "custom_word_poc" / "custom_word_scores.csv"

# Load trial scores
scores = []
labels = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        scores.append(float(row["score"]))
        labels.append(int(row["label"]))
scores = np.array(scores); labels = np.array(labels)

eer_pct = SUM["evaluation"]["EER_percent"]
eer_thr = SUM["evaluation"]["EER_threshold"]
mean_g  = SUM["evaluation"]["mean_genuine_score"]
mean_i  = SUM["evaluation"]["mean_imposter_score"]
tar1    = SUM["evaluation"]["TAR_at_FAR_1pct"]
tar5    = SUM["evaluation"]["TAR_at_FAR_5pct"]

per_word = SUM["per_word"]

# ROC
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(labels, scores)
auc = float(np.trapz(tpr, fpr))

# ---- plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

# (a) score histogram
ax = axes[0]
g = scores[labels == 1]; i = scores[labels == 0]
ax.hist(i, bins=40, alpha=0.65, color="#B91C1C", label=f"Imposter  (μ={mean_i:.3f})")
ax.hist(g, bins=40, alpha=0.65, color="#1E88E5", label=f"Genuine   (μ={mean_g:.3f})")
ax.axvline(eer_thr, color="#0C2340", linestyle="--", lw=1.2,
           label=f"EER threshold ({eer_thr:.2f})")
ax.set_xlabel("cosine similarity to 5-shot prototype")
ax.set_ylabel("count")
ax.set_title(f"(a) Held-out word scores  ·  EER = {eer_pct:.2f}%",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)

# (b) per-word EER
ax = axes[1]
words_sorted = sorted(per_word.items(), key=lambda x: x[1]["eer_percent"])
names = [w for w, _ in words_sorted]
eers  = [d["eer_percent"] for _, d in words_sorted]
colors = ["#65A30D" if e <= 10 else "#B45309" if e <= 15 else "#B91C1C" for e in eers]
bars = ax.bar(names, eers, color=colors, edgecolor="white")
for bar, v in zip(bars, eers):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.6, f"{v:.0f}",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(eer_pct, color="#0C2340", linestyle="--", lw=1.2,
           label=f"overall EER = {eer_pct:.2f}%")
ax.set_ylabel("per-word EER (%)")
ax.set_ylim(0, 22)
ax.set_title("(b) Per-novel-word EER  ·  green=easy, brown=mid, red=hard",
             fontsize=11, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

# (c) ROC + operating points
ax = axes[2]
ax.plot(fpr, tpr, color="#1428A0", lw=2.4, label=f"ROC  (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.5)
# Mark FAR=1% and FAR=5% operating points
for far_target, tar_value, color, name in [(0.01, tar1, "#B91C1C", "FAR=1%"),
                                            (0.05, tar5, "#B45309", "FAR=5%")]:
    ax.scatter([far_target], [tar_value], color=color, s=80, zorder=5,
               edgecolor="white", linewidth=1.5,
               label=f"{name}  →  TAR = {tar_value*100:.1f}%")
ax.set_xlabel("False Accept Rate")
ax.set_ylabel("True Accept Rate")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_title("(c) ROC — held-out custom-word detection",
             fontsize=11, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)

fig.suptitle(
    f"Custom-word few-shot detection  ·  {SUM['train_words']} train words → "
    f"{len(SUM['held_out_words'])} novel words enrolled with 5 clips each  "
    f"·  {SUM['model_params_k']} K params",
    fontsize=12, fontweight="bold", y=1.02,
)

plt.tight_layout()
plt.savefig(ROOT / "illustrations" / "10_custom_word.png",
            dpi=200, bbox_inches="tight")
plt.savefig(ROOT / "illustrations" / "10_custom_word.svg",
            bbox_inches="tight")
print(f"wrote 10_custom_word.{{png,svg}}  (overall EER {eer_pct:.2f}%, AUC {auc:.3f})")
