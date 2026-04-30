"""Step B v4 — FiLM with paired enrollment + hard negatives + weighted imposter loss.

Three changes over v3:
  1. HARD NEGATIVE MINING — for imposter pairs, pick the most-similar-embedding
     *different speaker* in the batch (not a random permutation). Forces FiLM
     to discriminate the hardest cases, not the average ones.
  2. WEIGHTED IMPOSTER LOSS — λ_imp = 2.0 so imposter mistakes hurt twice as
     much as genuine misclassifications. Pushes the model to favour rejection.
  3. HIGHER IMPOSTER RATIO — max_ratio = 0.7 (was 0.5). More rejection
     pressure at full ramp.

Expected: TAR ~85%, FAR ~25%, REJ ~70% — closes the v3 trade-off.

Usage from code/:
    .venv/bin/python scripts/film_poc_v4.py --device mps
    .venv/bin/python scripts/film_poc_v4.py --smoke --device mps
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
sys.path.insert(0, str(HERE))
# Reuse classes from v3 (same model architecture)
from film_poc_v3 import (
    JointKWSSV, SubCenterAAMSoftmax, SCPairedDataset,
    load_clip, load_p0_init, parse_speaker, imposter_ratio_for_epoch,
    SC_V2_WORDS, WORD_TO_IDX, UNKNOWN_LABEL, SC_ROOT, P0_CKPT,
    CLIP_LEN, SAMPLE_RATE,
)

REPO_ROOT = HERE.parent.parent
OUT_DIR   = REPO_ROOT / "results" / "film_poc_v4"


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
    ap.add_argument("--max_imposter_ratio", type=float, default=0.7,
                    help="v4 default 0.7 (v3 was 0.5)")
    ap.add_argument("--lambda_imposter", type=float, default=2.0,
                    help="v4 default 2.0 — upweights imposter misclassifications")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=2)
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

    # Speaker split
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
    print(f"train utts : {len(train_ds):,}")
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
    print(f"v4 changes : hard-negative mining + λ_imp={args.lambda_imposter}\n")

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

            trunk_feats = model.trunk_features(wav)
            sv_emb_own  = model.sv_embed(trunk_feats)

            with torch.no_grad():
                enroll_feats = model.trunk_features(enroll_wav)
                sv_emb_enroll = model.sv_embed(enroll_feats)

            if imp_ratio > 0:
                # ---------- v4 HARD NEGATIVE MINING ----------
                # For each sample, find the most-similar enrollment-emb from a
                # DIFFERENT speaker. argmax over (sim ∈ R^B) with same-speaker
                # entries masked to -inf.
                sim = sv_emb_enroll @ sv_emb_enroll.T          # (B, B)
                spk_eq = (spk.unsqueeze(0) == spk.unsqueeze(1)) # (B, B)
                sim = sim.masked_fill(spk_eq, -2.0)
                hard_neg_idx = sim.argmax(dim=1)                # (B,)

                is_imposter = (torch.rand(B, device=device) < imp_ratio)
                sv_for_film = torch.where(
                    is_imposter.unsqueeze(-1),
                    sv_emb_enroll[hard_neg_idx],   # hard imposter
                    sv_emb_enroll,                 # genuine
                )
                target = torch.where(is_imposter,
                                     torch.full_like(kw, UNKNOWN_LABEL), kw)
            else:
                is_imposter = torch.zeros(B, device=device, dtype=torch.bool)
                sv_for_film = sv_emb_enroll
                target = kw

            kws_logits = model.kws_logits(trunk_feats, sv_for_film)

            # ---------- v4 WEIGHTED LOSS ----------
            per_sample_kws = F.cross_entropy(kws_logits, target, reduction="none")
            weight = torch.where(
                is_imposter,
                torch.tensor(args.lambda_imposter, device=device, dtype=per_sample_kws.dtype),
                torch.tensor(1.0, device=device, dtype=per_sample_kws.dtype),
            )
            loss_kws = (per_sample_kws * weight).sum() / weight.sum()
            loss_sv  = aam_head(sv_emb_own, spk)
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
               OUT_DIR / "joint_kws_sv_film_v4.pt")
    print(f"\nsaved -> {OUT_DIR}/joint_kws_sv_film_v4.pt")

    # ---- eval (same enrolled-embedding methodology as v1/v2/v3)
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
        "version": "film_poc_v4 (paired + hard negatives + weighted imposter)",
        "device": str(device), "smoke": args.smoke,
        "trained_speakers": NUM_TRAIN_SPK,
        "training_utterances": len(train_ds),
        "epochs": args.epochs,
        "curriculum": {
            "ramp_start": args.ramp_start, "ramp_end": args.ramp_end,
            "max_imposter_ratio": args.max_imposter_ratio,
            "lambda_imposter": args.lambda_imposter,
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
        "v3_baseline": {
            "TAR_percent": 92.14, "FAR_percent": 48.21, "REJ_percent": 51.07,
            "collapse_ratio": 0.547,
        },
        "delta_vs_v3": {
            "TAR_pp": round(TAR * 100 - 92.14, 2),
            "FAR_pp": round(FAR * 100 - 48.21, 2),
            "REJ_pp": round(REJ * 100 - 51.07, 2),
        },
    }
    json.dump(summary, open(OUT_DIR / "film_summary_v4.json", "w"), indent=2)
    print()
    print(json.dumps(summary, indent=2))

    with open(OUT_DIR / "film_results_v4.csv", "w", newline="") as f:
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
    ax1.set_title("FiLM v4 (paired + hard neg + weighted)", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4); ax1.set_axisbelow(True)
    probs_g = [r["prob_true_kw"] for r in genuine_results]
    probs_i = [r["prob_true_kw"] for r in imposter_results]
    ax2.hist(probs_g, bins=30, alpha=0.6, color="#1E88E5",
             label=f"Genuine (mu={mean_prob_g:.3f})")
    ax2.hist(probs_i, bins=30, alpha=0.6, color="#B91C1C",
             label=f"Imposter (mu={mean_prob_i:.3f})")
    ax2.set_xlabel("P(true keyword)"); ax2.set_ylabel("count")
    ax2.set_title("Keyword-logit distribution v4", fontweight="bold")
    ax2.legend(); ax2.grid(linestyle="--", alpha=0.4); ax2.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "film_rejection_metrics_v4.png", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_DIR}/film_rejection_metrics_v4.png")


if __name__ == "__main__":
    main()
