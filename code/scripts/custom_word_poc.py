"""Step D — Custom-word detection POC via few-shot metric learning.

Trains a phonetic embedding network on 28 SC V2 words with Sub-center
AAM-Softmax. Evaluates by enrolling 7 HELD-OUT words (model never saw
them during training): for each held-out word, average 5 random clips
into a prototype, then score genuine (same word, different clips) vs
imposter (other held-out words) cos-similarities.

This is the few-shot "user-chosen trigger word" generalization test.
If it works at <600K params without a WavLM teacher, we have a clean
custom-word story. If not, v2 will add WavLM distillation.

Usage from code/:
    .venv/bin/python scripts/custom_word_poc.py --device mps
    .venv/bin/python scripts/custom_word_poc.py --smoke --device mps
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.model import BCResNet8, LearnablePCEN

# ---- word splits ----
TRAIN_WORDS = [
    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
    "bed", "bird", "cat", "dog", "happy", "house", "tree", "wow",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
]  # 28 words
HELD_OUT_WORDS = [
    "backward", "forward", "marvin", "sheila", "learn", "follow", "visual",
]  # 7 polysyllabic words — the model has never seen these

WORD_TO_IDX = {w: i for i, w in enumerate(TRAIN_WORDS)}
SAMPLE_RATE = 16000
CLIP_LEN = 16000

REPO_ROOT = HERE.parent.parent
SC_ROOT   = REPO_ROOT / "data" / "speech_commands_v2"
P0_CKPT   = REPO_ROOT / "results (1)" / "runs" / "p0" / "best.pt"
OUT_DIR   = REPO_ROOT / "results" / "custom_word_poc"


def parse_speaker(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def load_clip(path: str) -> torch.Tensor:
    wav, _ = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    if len(wav) < CLIP_LEN:
        wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
    return torch.from_numpy(wav[:CLIP_LEN])


# ---------------------------------------------------------------- dataset
class SCWordDataset(Dataset):
    """Yields (wav, word_idx). One row per training utterance."""
    def __init__(self, root, words, max_per_word=400, seed=1337):
        rng = random.Random(seed)
        items = []
        for w in words:
            wavs = list((root / w).glob("*.wav"))
            rng.shuffle(wavs)
            for wp in wavs[:max_per_word]:
                items.append((str(wp), WORD_TO_IDX[w]))
        rng.shuffle(items)
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav, _ = sf.read(path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = wav.astype(np.float32)
        if len(wav) < CLIP_LEN:
            wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
        elif len(wav) > CLIP_LEN:
            wav = wav[:CLIP_LEN]
        return torch.from_numpy(wav), label


# ---------------------------------------------------------------- model
class MelFrontend(nn.Module):
    def __init__(self, n_mels=40, sample_rate=16000, use_pcen=True):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=512, win_length=400,
            hop_length=160, n_mels=n_mels, f_min=20.0,
            f_max=sample_rate // 2, power=2.0,
        )
        self.use_pcen = use_pcen
        if use_pcen:
            self.pcen = LearnablePCEN(n_mels)
        self.eps = 1e-6

    def forward(self, wav):
        mel = self.melspec(wav)
        return self.pcen(mel) if self.use_pcen else torch.log(mel + self.eps)


class AttnStatPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Conv1d(in_dim, in_dim, 1)

    def forward(self, x):
        alpha = F.softmax(self.attn(x), dim=-1)
        mean  = (alpha * x).sum(dim=-1)
        var   = (alpha * x ** 2).sum(dim=-1) - mean ** 2
        std   = torch.sqrt(var.clamp(min=1e-6))
        return torch.cat([mean, std], dim=-1)


class WordEmbedder(nn.Module):
    """Trunk + pool + projection → L2-normalized 256-dim phonetic embedding."""
    def __init__(self, n_mels=40, emb_dim=256, use_pcen=True):
        super().__init__()
        self.frontend = MelFrontend(n_mels, use_pcen=use_pcen)
        self.trunk    = BCResNet8(n_mels=n_mels)
        C = self.trunk.out_channels
        self.pool     = AttnStatPool(C)
        self.proj     = nn.Linear(2 * C, emb_dim)

    def forward(self, wav):
        spec  = self.frontend(wav)
        feats = self.trunk(spec.unsqueeze(1))
        feats = feats.mean(dim=2)             # (B, C, T'')
        pooled = self.pool(feats)             # (B, 2C)
        emb    = self.proj(pooled)
        return F.normalize(emb, dim=-1)


class SubCenterAAMSoftmax(nn.Module):
    def __init__(self, emb_dim, num_classes, K=3, margin=0.2, scale=30.0):
        super().__init__()
        self.num_classes = num_classes
        self.K = K
        self.m = margin
        self.s = scale
        self.W = nn.Parameter(torch.empty(num_classes * K, emb_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, emb, label):
        W_norm = F.normalize(self.W, dim=-1)
        cos_all = emb @ W_norm.T
        cos_all = cos_all.view(-1, self.num_classes, self.K)
        cos     = cos_all.max(dim=-1).values
        B = cos.size(0)
        idx_b = torch.arange(B, device=cos.device)
        target_cos = cos[idx_b, label].clamp(-1 + 1e-7, 1 - 1e-7)
        sin = torch.sqrt(1 - target_cos ** 2)
        cos_m = target_cos * math.cos(self.m) - sin * math.sin(self.m)
        logits = cos.clone() * self.s
        logits[idx_b, label] = cos_m * self.s
        return F.cross_entropy(logits, label), cos


def load_p0_init(model: WordEmbedder, ckpt_path: Path) -> int:
    if not ckpt_path.is_file():
        print(f"  WARN: P0 ckpt not found — random init")
        return 0
    p0 = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    p0_state = p0["model"]
    own_state = model.state_dict()
    transferred = []
    for k, v in p0_state.items():
        if (k.startswith("frontend.") or k.startswith("trunk.")) and k in own_state:
            if own_state[k].shape == v.shape:
                own_state[k] = v
                transferred.append(k)
    model.load_state_dict(own_state)
    print(f"  transferred {len(transferred)} P0 tensors")
    return len(transferred)


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer, float(thresholds[idx]), fpr, tpr, thresholds


def tar_at_far(scores, labels, target_far=0.01):
    """Compute TAR (true accept rate on label==1) at the threshold where FAR == target_far."""
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0 or len(pos_scores) == 0:
        return None, None
    threshold = float(np.quantile(neg_scores, 1 - target_far))
    tar = float((pos_scores >= threshold).mean())
    return tar, threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=[None, "cpu", "mps", "cuda"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max_per_word", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--enroll_per_word", type=int, default=5,
                    help="number of clips averaged into the held-out prototype")
    ap.add_argument("--test_per_word", type=int, default=50,
                    help="number of genuine test clips per held-out word")
    ap.add_argument("--p0_ckpt", default=str(P0_CKPT))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.smoke:
        args.max_per_word = 30
        args.epochs = 5
        args.test_per_word = 15
        print("[smoke mode] 30 utts/word, 5 epochs")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device     : {device}")
    print(f"output dir : {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify both word groups exist as folders
    missing_train = [w for w in TRAIN_WORDS if not (SC_ROOT / w).is_dir()]
    missing_held  = [w for w in HELD_OUT_WORDS if not (SC_ROOT / w).is_dir()]
    if missing_train or missing_held:
        print(f"  WARN: missing word folders: train={missing_train} held={missing_held}")
    print(f"train words ({len(TRAIN_WORDS)}): {TRAIN_WORDS}")
    print(f"held-out words ({len(HELD_OUT_WORDS)}): {HELD_OUT_WORDS}")

    # ---- training data
    train_ds = SCWordDataset(SC_ROOT, TRAIN_WORDS,
                             max_per_word=args.max_per_word, seed=args.seed)
    print(f"train utts : {len(train_ds):,}")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # ---- model
    model = WordEmbedder(emb_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model      : {n_params/1e3:.1f}K params")
    print(f"P0 init    :")
    load_p0_init(model, Path(args.p0_ckpt))
    model = model.to(device)

    aam_head = SubCenterAAMSoftmax(emb_dim=256, num_classes=len(TRAIN_WORDS),
                                    K=3, margin=0.2, scale=30.0).to(device)
    print(f"AAM head   : {sum(p.numel() for p in aam_head.parameters())/1e3:.1f}K (training-only)")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aam_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs,
    )

    # ---- training
    print()
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train(); aam_head.train()
        loss_sum = correct = n = 0
        for wav, label in train_loader:
            wav = wav.to(device); label = label.to(device)
            emb = model(wav)
            loss, cos = aam_head(emb, label)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                pred = cos.argmax(dim=-1)
                correct += (pred == label).sum().item()
            loss_sum += loss.item() * wav.size(0)
            n += wav.size(0)
        dt = time.time() - t0
        acc = correct / n
        history.append({"epoch": epoch, "loss": loss_sum/n, "acc": acc, "wall_s": dt})
        print(f"ep{epoch:>2} | loss={loss_sum/n:.3f} | train_word_acc={acc:.3f} | {dt:.1f}s")

    torch.save({"model": model.state_dict()}, OUT_DIR / "word_embedder.pt")
    print(f"\nsaved -> {OUT_DIR}/word_embedder.pt")

    # ---- eval: few-shot enrollment of held-out words
    model.eval()
    rng = random.Random(2026)

    held_word_clips = {w: list((SC_ROOT / w).glob("*.wav")) for w in HELD_OUT_WORDS}
    for w, clips in held_word_clips.items():
        rng.shuffle(clips)
        print(f"  held-out {w}: {len(clips)} clips available")

    # Build prototypes (5-shot per word)
    prototypes = {}
    test_clips = {}     # genuine pool per word
    used_for_enroll = set()
    with torch.no_grad():
        for w, clips in held_word_clips.items():
            if len(clips) < args.enroll_per_word + args.test_per_word:
                print(f"  SKIP {w}: only {len(clips)} clips")
                continue
            enroll = clips[:args.enroll_per_word]
            test   = clips[args.enroll_per_word:args.enroll_per_word + args.test_per_word]
            for c in enroll:
                used_for_enroll.add(str(c))
            wavs = torch.stack([load_clip(str(p)) for p in enroll]).to(device)
            embs = model(wavs)
            prototypes[w] = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)
            test_clips[w] = test

    print(f"\nprototypes ready for {len(prototypes)} held-out words")

    # Build trial pairs:
    # genuine = (test_clip of word w, prototype of w)
    # imposter = (test_clip of word w', prototype of w) where w != w'
    scores, labels, meta = [], [], []
    rng2 = random.Random(2026)
    held_words_with_proto = list(prototypes.keys())
    with torch.no_grad():
        for w, tests in test_clips.items():
            for clip_path in tests:
                wav = load_clip(str(clip_path)).unsqueeze(0).to(device)
                emb = model(wav)
                # Genuine: this clip vs its own word's prototype
                proto = prototypes[w]
                s_gen = float((emb @ proto.T).item())
                scores.append(s_gen); labels.append(1); meta.append({"true_word": w, "proto_word": w})
                # Imposter: this clip vs another held-out word's prototype
                other_w = rng2.choice([x for x in held_words_with_proto if x != w])
                s_imp = float((emb @ prototypes[other_w].T).item())
                scores.append(s_imp); labels.append(0); meta.append({"true_word": w, "proto_word": other_w})

    scores = np.array(scores); labels = np.array(labels)
    print(f"trial pairs: {len(scores):,} ({(labels==1).sum()} genuine, {(labels==0).sum()} imposter)")

    eer, eer_thr, fpr, tpr, thresholds = compute_eer(scores, labels)
    tar_at_1pct, thr_at_1pct = tar_at_far(scores, labels, target_far=0.01)
    tar_at_5pct, thr_at_5pct = tar_at_far(scores, labels, target_far=0.05)
    mean_gen = float(scores[labels == 1].mean())
    mean_imp = float(scores[labels == 0].mean())

    # Per-word EER for diagnostics
    per_word = {}
    for w in held_words_with_proto:
        # sub-select trials where the test clip's true word is w
        keep = np.array([m["true_word"] == w for m in meta])
        if keep.sum() < 4:
            continue
        s_w, l_w = scores[keep], labels[keep]
        if l_w.sum() == 0 or l_w.sum() == len(l_w):
            continue
        eer_w, _, _, _, _ = compute_eer(s_w, l_w)
        per_word[w] = {
            "eer_percent": round(eer_w * 100, 2),
            "n_pairs": int(keep.sum()),
            "mean_genuine": float(s_w[l_w == 1].mean()),
            "mean_imposter": float(s_w[l_w == 0].mean()),
        }

    summary = {
        "step": "D — custom-word few-shot detection",
        "device": str(device), "smoke": args.smoke,
        "train_words": len(TRAIN_WORDS),
        "train_utterances": len(train_ds),
        "held_out_words": list(prototypes.keys()),
        "epochs": args.epochs,
        "model_params_k": round(n_params / 1e3, 1),
        "evaluation": {
            "trial_pairs": int(len(scores)),
            "EER_percent": round(eer * 100, 2),
            "EER_threshold": round(eer_thr, 4),
            "TAR_at_FAR_1pct": round(tar_at_1pct, 4) if tar_at_1pct is not None else None,
            "TAR_at_FAR_5pct": round(tar_at_5pct, 4) if tar_at_5pct is not None else None,
            "mean_genuine_score": round(mean_gen, 4),
            "mean_imposter_score": round(mean_imp, 4),
            "score_separation": round(mean_gen - mean_imp, 4),
        },
        "per_word": per_word,
    }
    json.dump(summary, open(OUT_DIR / "custom_word_summary.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))

    with open(OUT_DIR / "custom_word_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true_word", "proto_word", "label", "score"])
        for s, l, m in zip(scores.tolist(), labels.tolist(), meta):
            w.writerow([m["true_word"], m["proto_word"], l, s])

    # Plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) Genuine vs imposter score histogram
    ax = axes[0]
    g = scores[labels == 1]; i = scores[labels == 0]
    ax.hist(i, bins=40, alpha=0.6, color="#B91C1C", label=f"Imposter (mu={mean_imp:.3f})")
    ax.hist(g, bins=40, alpha=0.6, color="#1E88E5", label=f"Genuine  (mu={mean_gen:.3f})")
    ax.axvline(eer_thr, color="red", linestyle="--", lw=1, label=f"EER threshold ({eer_thr:.3f})")
    ax.set_xlabel("cosine similarity to prototype")
    ax.set_ylabel("count")
    ax.set_title(f"Held-out word scores  (EER = {eer*100:.2f}%)", fontweight="bold")
    ax.legend(); ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)

    # (b) Per-word EER bar
    ax = axes[1]
    words_sorted = sorted(per_word.items(), key=lambda x: x[1]["eer_percent"])
    names = [w for w, _ in words_sorted]
    eers  = [d["eer_percent"] for _, d in words_sorted]
    bars = ax.bar(names, eers, color="#65A30D", edgecolor="white")
    for bar, v in zip(bars, eers):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(eer * 100, color="red", linestyle="--", lw=1,
               label=f"overall EER = {eer*100:.2f}%")
    ax.set_ylabel("per-word EER (%)")
    ax.set_title("Per-held-out-word EER  (lower = easier)", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # (c) ROC curve
    ax = axes[2]
    ax.plot(fpr, tpr, color="#1E88E5", lw=2, label=f"AUC = {np.trapz(tpr, fpr):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
    ax.scatter([fpr[np.argmin(np.abs(fpr - (1 - tpr)))]],
               [tpr[np.argmin(np.abs(fpr - (1 - tpr)))]],
               color="red", s=60, zorder=5, label=f"EER point")
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title("ROC — held-out custom-word detection", fontweight="bold")
    ax.legend(); ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "custom_word_metrics.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/custom_word_metrics.png")

    print()
    print("Go / no-go (custom-word few-shot):")
    print("  EER <= 10%  GREEN  -> custom-word generalization works at <600K params")
    print("  EER 10-20%  YELLOW -> works but needs WavLM distillation for production")
    print("  EER 20-30%  ORANGE -> WavLM distillation required (Step D v2)")
    print("  EER > 30%   RED    -> architecture limit; rethink phonetic representation")
    print(f"\nAchieved: EER = {eer*100:.2f}%")


if __name__ == "__main__":
    main()
