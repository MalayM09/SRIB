"""Step B v3 — FiLM with PAIRED-ENROLLMENT training.

The key change vs v2: at training time, the FiLM modulator is computed from a
*different clip of the same speaker* (genuine) or a *different speaker's clip*
(imposter), NOT from the audio being classified. This matches the deployment
scenario where the enrollment embedding and the test audio are different clips.

Closes the train/eval distribution gap that capped v2 at TAR=62.5% on enrolled
eval (despite the trained model achieving 94.6% TAR with own-audio embeddings —
see film_v2_diagnostic.py).

Usage from code/:
    .venv/bin/python scripts/film_poc_v3.py --device mps
    .venv/bin/python scripts/film_poc_v3.py --smoke --device mps
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

SC_V2_WORDS = ["yes", "no", "up", "down", "left",
               "right", "on", "off", "stop", "go"]
WORD_TO_IDX = {w: i for i, w in enumerate(SC_V2_WORDS)}
UNKNOWN_LABEL = 10
SAMPLE_RATE = 16000
CLIP_LEN = 16000

REPO_ROOT = HERE.parent.parent
SC_ROOT   = REPO_ROOT / "data" / "speech_commands_v2"
P0_CKPT   = REPO_ROOT / "results (1)" / "runs" / "p0" / "best.pt"
OUT_DIR   = REPO_ROOT / "results" / "film_poc_v3"


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
class SCPairedDataset(Dataset):
    """For each (test_audio, kw, spk), also samples a different enrollment clip
    from the same speaker. The enrollment audio drives FiLM at training."""

    def __init__(self, root, speakers_allowed, speaker_to_idx,
                 max_per_speaker=50, seed=1337):
        rng = random.Random(seed)
        per_spk = defaultdict(list)
        for word in SC_V2_WORDS:
            for wp in (root / word).glob("*.wav"):
                spk = parse_speaker(wp.name)
                if spk not in speakers_allowed:
                    continue
                per_spk[spk].append((str(wp), WORD_TO_IDX[word]))

        # Cap, then keep speakers with >= 2 clips (need pair)
        items = []
        per_spk_capped = {}
        for spk, lst in per_spk.items():
            if len(lst) < 2:
                continue
            rng.shuffle(lst)
            capped = lst[:max_per_speaker]
            per_spk_capped[spk] = capped
            for path, kw in capped:
                items.append((path, kw, spk))
        rng.shuffle(items)
        self.items = items
        self.per_spk = per_spk_capped
        self.speaker_to_idx = speaker_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, kw, spk = self.items[idx]
        # Pick a DIFFERENT clip from same speaker as the FiLM enrollment
        spk_clips = self.per_spk[spk]
        # np.random instead of random.choice — works with DataLoader workers
        i_enroll = int(np.random.randint(0, len(spk_clips)))
        if spk_clips[i_enroll][0] == path and len(spk_clips) > 1:
            i_enroll = (i_enroll + 1) % len(spk_clips)
        enroll_path = spk_clips[i_enroll][0]

        wav = load_clip(path)
        enroll_wav = load_clip(enroll_path)
        return wav, enroll_wav, kw, self.speaker_to_idx[spk]


# ---------------------------------------------------------------- model (same as v2)
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


class JointKWSSV(nn.Module):
    def __init__(self, n_mels=40, num_keywords=12, sv_dim=256, use_pcen=True):
        super().__init__()
        self.frontend = MelFrontend(n_mels, use_pcen=use_pcen)
        self.trunk    = BCResNet8(n_mels=n_mels)
        C = self.trunk.out_channels
        self.sv_pool  = AttnStatPool(C)
        self.sv_proj  = nn.Linear(2 * C, sv_dim)
        self.kws_pool = AttnStatPool(C)
        self.film_gen = nn.Linear(sv_dim, 2 * 2 * C)
        self.kws_head = nn.Linear(2 * C, num_keywords)

    def trunk_features(self, wav):
        spec  = self.frontend(wav)
        feats = self.trunk(spec.unsqueeze(1))
        return feats.mean(dim=2)

    def sv_embed(self, trunk_feats):
        sv_pooled = self.sv_pool(trunk_feats)
        return F.normalize(self.sv_proj(sv_pooled), dim=-1)

    def kws_logits(self, trunk_feats, sv_emb_modulator):
        kws_pooled = self.kws_pool(trunk_feats)
        film  = self.film_gen(sv_emb_modulator)
        gamma, beta = film.chunk(2, dim=-1)
        modulated = kws_pooled * (1 + gamma) + beta
        return self.kws_head(modulated)


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
        return F.cross_entropy(logits, label)


def load_p0_init(model: JointKWSSV, ckpt_path: Path) -> int:
    if not ckpt_path.is_file():
        print(f"  WARN: P0 ckpt not found at {ckpt_path} — random init")
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
    print(f"  transferred {len(transferred)} tensors from P0")
    return len(transferred)


def imposter_ratio_for_epoch(epoch, ramp_start, ramp_end, max_ratio=0.5):
    if epoch <= ramp_start:
        return 0.0
    if epoch >= ramp_end:
        return max_ratio
    return max_ratio * (epoch - ramp_start) / (ramp_end - ramp_start)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=[None, "cpu", "mps", "cuda"], default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_train_spk", type=int, default=400)
    ap.add_argument("--n_eval_spk",  type=int, default=80)
    ap.add_argument("--max_per_speaker", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--ramp_start", type=int, default=5)
    ap.add_argument("--ramp_end", type=int, default=15)
    ap.add_argument("--max_imposter_ratio", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--p0_ckpt", default=str(P0_CKPT))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.smoke:
        args.n_train_spk = 50
        args.n_eval_spk = 15
        args.max_per_speaker = 15
        args.epochs = 6
        args.ramp_start = 2
        args.ramp_end = 4
        print("[smoke mode]")

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

    counts = Counter()
    for word in SC_V2_WORDS:
        for wp in (SC_ROOT / word).glob("*.wav"):
            counts[parse_speaker(wp.name)] += 1
    eligible = sorted(s for s, n in counts.items() if n >= 8)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    train_spks = set(eligible[:args.n_train_spk])
    eval_spks  = set(eligible[args.n_train_spk : args.n_train_spk + args.n_eval_spk])
    train_speaker_to_idx = {s: i for i, s in enumerate(sorted(train_spks))}
    NUM_TRAIN_SPK = len(train_speaker_to_idx)
    print(f"speakers   : train={NUM_TRAIN_SPK}  eval={len(eval_spks)}")

    train_ds = SCPairedDataset(
        root=SC_ROOT, speakers_allowed=train_spks,
        speaker_to_idx=train_speaker_to_idx,
        max_per_speaker=args.max_per_speaker, seed=args.seed,
    )
    print(f"train utts : {len(train_ds):,}  (each yields a paired enrollment clip)")
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model = JointKWSSV(num_keywords=12, sv_dim=256).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model      : {n_params/1e3:.1f}K params")
    print(f"P0 init    :")
    load_p0_init(model, Path(args.p0_ckpt))
    model = model.to(device)
    aam_head = SubCenterAAMSoftmax(emb_dim=256, num_classes=NUM_TRAIN_SPK, K=3).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(aam_head.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs,
    )

    print(f"\ncurriculum : imposter_ratio = 0 for ep<={args.ramp_start}, "
          f"linear ramp to {args.max_imposter_ratio} by ep{args.ramp_end}")
    print(f"FiLM input : ALWAYS from a DIFFERENT clip of (same|different) speaker")

    history = []
    for epoch in range(1, args.epochs + 1):
        imp_ratio = imposter_ratio_for_epoch(
            epoch, args.ramp_start, args.ramp_end, args.max_imposter_ratio,
        )
        t0 = time.time()
        model.train(); aam_head.train()
        lk_sum = ls_sum = 0.0
        gen_correct = imp_rej = n_gen = n_imp = 0
        for wav, enroll_wav, kw, spk in train_loader:
            wav = wav.to(device); enroll_wav = enroll_wav.to(device)
            kw = kw.to(device);   spk = spk.to(device)
            B = wav.size(0)

            # Forward: test audio
            trunk_feats = model.trunk_features(wav)
            sv_emb_test = model.sv_embed(trunk_feats)   # for AAM loss only

            # Forward: enrollment audio (different clip, same speaker → genuine FiLM)
            with torch.no_grad():
                enroll_feats = model.trunk_features(enroll_wav)
                sv_emb_enroll = model.sv_embed(enroll_feats)
            # Note: detached implicitly by torch.no_grad() — FiLM gen still gets gradients,
            # but no gradient flows through the enrollment trunk pass.

            if imp_ratio > 0:
                is_imposter = (torch.rand(B, device=device) < imp_ratio)
                perm = torch.randperm(B, device=device)
                # Imposter FiLM: another batch sample's enrollment (different speaker)
                sv_for_film = torch.where(
                    is_imposter.unsqueeze(-1),
                    sv_emb_enroll[perm],
                    sv_emb_enroll,
                )
                target = torch.where(is_imposter,
                                     torch.full_like(kw, UNKNOWN_LABEL), kw)
            else:
                is_imposter = torch.zeros(B, device=device, dtype=torch.bool)
                sv_for_film = sv_emb_enroll
                target = kw

            kws_logits = model.kws_logits(trunk_feats, sv_for_film)
            loss_kws = F.cross_entropy(kws_logits, target)
            loss_sv  = aam_head(sv_emb_test, spk)
            loss = loss_kws + loss_sv

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = kws_logits.argmax(dim=-1)
                gen_mask = ~is_imposter
                gen_correct += ((preds == target) & gen_mask).sum().item()
                if is_imposter.any():
                    imp_rej += ((preds == UNKNOWN_LABEL) & is_imposter).sum().item()
                n_gen += int(gen_mask.sum().item())
                n_imp += int(is_imposter.sum().item())
            lk_sum += loss_kws.item() * B
            ls_sum += loss_sv.item()  * B

        n = n_gen + n_imp
        dt = time.time() - t0
        gen_acc = gen_correct / max(1, n_gen)
        imp_rej_rate = (imp_rej / n_imp) if n_imp > 0 else float("nan")
        history.append({
            "epoch": epoch, "imp_ratio": imp_ratio,
            "loss_kws": lk_sum/n, "loss_sv": ls_sum/n,
            "gen_acc": gen_acc, "imp_rej": imp_rej_rate, "wall_s": dt,
        })
        imp_str = f"imp_rej={imp_rej_rate:.3f}" if n_imp > 0 else "imp_rej=  -  "
        print(f"ep{epoch:>2} | imp_r={imp_ratio:.2f} | "
              f"kws={lk_sum/n:.3f} | sv={ls_sum/n:.3f} "
              f"| gen_acc={gen_acc:.3f} | {imp_str} | {dt:.1f}s")

    torch.save({"model": model.state_dict(), "aam": aam_head.state_dict()},
               OUT_DIR / "joint_kws_sv_film_v3.pt")
    print(f"\nsaved -> {OUT_DIR}/joint_kws_sv_film_v3.pt")

    # ---- eval (same enrolled-embedding methodology as v1/v2)
    eval_per_spk = defaultdict(list)
    for word in SC_V2_WORDS:
        for wp in (SC_ROOT / word).glob("*.wav"):
            spk = parse_speaker(wp.name)
            if spk in eval_spks:
                eval_per_spk[spk].append((str(wp), WORD_TO_IDX[word]))

    ENROLL_PER_SPK = 5
    TEST_PER_SPK   = 10
    enroll_items = {}; test_items = []
    rng = random.Random(2026)
    for spk, lst in eval_per_spk.items():
        if len(lst) < ENROLL_PER_SPK + TEST_PER_SPK:
            continue
        rng.shuffle(lst)
        enroll_items[spk] = lst[:ENROLL_PER_SPK]
        for path, kw in lst[ENROLL_PER_SPK:ENROLL_PER_SPK + TEST_PER_SPK]:
            test_items.append({"speaker": spk, "path": path, "keyword": kw})
    print(f"eval       : {len(enroll_items)} enrolled, {len(test_items)} test items")

    model.eval()
    enroll_embs = {}
    with torch.no_grad():
        for spk, items in enroll_items.items():
            wavs = torch.stack([load_clip(p) for p, _ in items]).to(device)
            feats = model.trunk_features(wavs)
            embs  = model.sv_embed(feats)
            enroll_embs[spk] = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)

    genuine_results = []; imposter_results = []
    speaker_list = list(enroll_embs.keys())
    rng2 = random.Random(2026)
    with torch.no_grad():
        for ti in test_items:
            wav = load_clip(ti["path"]).unsqueeze(0).to(device)
            feats = model.trunk_features(wav)
            true_spk = ti["speaker"]; true_kw = ti["keyword"]

            logits_g = model.kws_logits(feats, enroll_embs[true_spk])
            pred_g   = int(logits_g.argmax(dim=-1).item())
            prob_g   = float(F.softmax(logits_g, dim=-1)[0, true_kw].item())
            genuine_results.append({"true_kw": true_kw, "pred_kw": pred_g,
                                    "prob_true_kw": prob_g})

            imposter_spk = rng2.choice([s for s in speaker_list if s != true_spk])
            logits_i = model.kws_logits(feats, enroll_embs[imposter_spk])
            pred_i   = int(logits_i.argmax(dim=-1).item())
            prob_i   = float(F.softmax(logits_i, dim=-1)[0, true_kw].item())
            imposter_results.append({"true_kw": true_kw, "pred_kw": pred_i,
                                     "prob_true_kw": prob_i})

    gen_correct = sum(1 for r in genuine_results if r["pred_kw"] == r["true_kw"])
    TAR = gen_correct / len(genuine_results)
    imp_falseaccept = sum(1 for r in imposter_results if r["pred_kw"] == r["true_kw"])
    FAR = imp_falseaccept / len(imposter_results)
    REJ = sum(1 for r in imposter_results if r["pred_kw"] == UNKNOWN_LABEL) / len(imposter_results)
    mean_prob_g = sum(r["prob_true_kw"] for r in genuine_results) / len(genuine_results)
    mean_prob_i = sum(r["prob_true_kw"] for r in imposter_results) / len(imposter_results)

    summary = {
        "version": "film_poc_v3 (paired-enrollment training)",
        "device": str(device), "smoke": args.smoke,
        "trained_speakers": NUM_TRAIN_SPK,
        "training_utterances": len(train_ds),
        "epochs": args.epochs,
        "curriculum": {
            "ramp_start": args.ramp_start, "ramp_end": args.ramp_end,
            "max_imposter_ratio": args.max_imposter_ratio,
        },
        "lr": args.lr,
        "model_params_k": round(n_params / 1e3, 1),
        "evaluation": {
            "enrolled_speakers": len(enroll_embs),
            "test_items": len(test_items),
            "genuine_TAR_percent": round(TAR * 100, 2),
            "imposter_FAR_percent": round(FAR * 100, 2),
            "imposter_rejection_percent": round(REJ * 100, 2),
            "mean_prob_true_kw_genuine": round(mean_prob_g, 4),
            "mean_prob_true_kw_imposter": round(mean_prob_i, 4),
            "probability_collapse_ratio": round(mean_prob_i / max(mean_prob_g, 1e-6), 4),
        },
        "v2_baseline_enrolled_eval": {
            "TAR_percent": 62.50, "FAR_percent": 15.00, "REJ_percent": 85.00,
            "collapse_ratio": 0.336,
        },
        "v2_diagnostic_own_audio": {
            "TAR_percent": 94.64, "FAR_percent": 14.29, "REJ_percent": 85.71,
            "collapse_ratio": 0.199,
        },
        "delta_vs_v2_enrolled": {
            "TAR_pp": round(TAR * 100 - 62.50, 2),
            "FAR_pp": round(FAR * 100 - 15.00, 2),
            "REJ_pp": round(REJ * 100 - 85.00, 2),
        },
    }
    json.dump(summary, open(OUT_DIR / "film_summary_v3.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))

    with open(OUT_DIR / "film_results_v3.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_type", "true_kw", "pred_kw", "prob_true_kw"])
        for r in genuine_results:
            w.writerow(["genuine", r["true_kw"], r["pred_kw"], r["prob_true_kw"]])
        for r in imposter_results:
            w.writerow(["imposter", r["true_kw"], r["pred_kw"], r["prob_true_kw"]])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(["Genuine\nTAR", "Imposter\nFAR", "Imposter\nrejection"],
            [TAR*100, FAR*100, REJ*100],
            color=["#1E88E5", "#B91C1C", "#65A30D"], edgecolor="white")
    for i, v in enumerate([TAR, FAR, REJ]):
        ax1.text(i, v*100 + 1.5, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax1.set_ylabel("rate (%)"); ax1.set_ylim(0, 105)
    ax1.set_title("FiLM v3 (paired enrollment training)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
    probs_g = [r["prob_true_kw"] for r in genuine_results]
    probs_i = [r["prob_true_kw"] for r in imposter_results]
    ax2.hist(probs_g, bins=30, alpha=0.6, color="#1E88E5",
             label=f"Genuine (mu={mean_prob_g:.3f})")
    ax2.hist(probs_i, bins=30, alpha=0.6, color="#B91C1C",
             label=f"Imposter (mu={mean_prob_i:.3f})")
    ax2.set_xlabel("P(true keyword)"); ax2.set_ylabel("count")
    ax2.set_title("Keyword-logit distribution v3", fontweight="bold")
    ax2.legend(); ax2.grid(linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "film_rejection_metrics_v3.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/film_rejection_metrics_v3.png")


if __name__ == "__main__":
    main()
