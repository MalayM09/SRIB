"""Diagnostic for FiLM v2 — compare TAR using enrolled vs own-audio embedding.

If TAR (own-audio) >> TAR (enrolled), the train/eval distribution mismatch
is confirmed and v3 with paired-enrollment training would close the gap.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from film_poc_v2 import (  # noqa: E402
    JointKWSSV, parse_speaker, load_clip,
    SC_V2_WORDS, WORD_TO_IDX, UNKNOWN_LABEL, SC_ROOT, OUT_DIR,
)

CKPT_DEFAULT = Path("/Users/malaymishra/Desktop/srib/results/film_poc_v2/joint_kws_sv_film_v2.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ckpt", default=str(CKPT_DEFAULT))
    ap.add_argument("--n_train_spk", type=int, default=400)
    ap.add_argument("--n_eval_spk",  type=int, default=80)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")

    model = JointKWSSV(num_keywords=12, sv_dim=256).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded: {Path(args.ckpt).name}")

    # Rebuild eval speaker split (must match training seed)
    counts: Counter = Counter()
    for word in SC_V2_WORDS:
        for wp in (SC_ROOT / word).glob("*.wav"):
            counts[parse_speaker(wp.name)] += 1
    eligible = sorted(s for s, n in counts.items() if n >= 8)
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    eval_spks = set(eligible[args.n_train_spk : args.n_train_spk + args.n_eval_spk])

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
    print(f"eval: {len(enroll_items)} enrolled, {len(test_items)} test items")

    # Pre-compute enrolled embeddings (5-clip avg, current eval method)
    enroll_embs = {}
    with torch.no_grad():
        for spk, items in enroll_items.items():
            wavs = torch.stack([load_clip(p) for p, _ in items]).to(device)
            feats = model.trunk_features(wavs)
            embs  = model.sv_embed(feats)
            enroll_embs[spk] = F.normalize(embs.mean(dim=0, keepdim=True), dim=-1)

    speaker_list = list(enroll_embs.keys())
    rng2 = random.Random(2026)
    results_enrolled = []
    results_own = []

    with torch.no_grad():
        for ti in test_items:
            wav = load_clip(ti["path"]).unsqueeze(0).to(device)
            feats = model.trunk_features(wav)
            true_spk = ti["speaker"]
            true_kw  = ti["keyword"]

            sv_emb_own = model.sv_embed(feats)
            imposter_spk = rng2.choice([s for s in speaker_list if s != true_spk])

            # ----- enrolled-embedding eval (current method)
            logits_g_e = model.kws_logits(feats, enroll_embs[true_spk])
            pred_g_e   = int(logits_g_e.argmax(dim=-1).item())
            prob_g_e   = float(F.softmax(logits_g_e, dim=-1)[0, true_kw].item())

            logits_i_e = model.kws_logits(feats, enroll_embs[imposter_spk])
            pred_i_e   = int(logits_i_e.argmax(dim=-1).item())
            prob_i_e   = float(F.softmax(logits_i_e, dim=-1)[0, true_kw].item())

            # ----- own-audio embedding eval (training-distribution)
            logits_g_o = model.kws_logits(feats, sv_emb_own)
            pred_g_o   = int(logits_g_o.argmax(dim=-1).item())
            prob_g_o   = float(F.softmax(logits_g_o, dim=-1)[0, true_kw].item())

            # Imposter: use a single clip's own embedding from another speaker
            imp_clip_path, _ = enroll_items[imposter_spk][0]
            imp_wav = load_clip(imp_clip_path).unsqueeze(0).to(device)
            imp_feats = model.trunk_features(imp_wav)
            sv_emb_imposter_own = model.sv_embed(imp_feats)
            logits_i_o = model.kws_logits(feats, sv_emb_imposter_own)
            pred_i_o   = int(logits_i_o.argmax(dim=-1).item())
            prob_i_o   = float(F.softmax(logits_i_o, dim=-1)[0, true_kw].item())

            results_enrolled.append({
                "true_kw": true_kw,
                "gen_pred": pred_g_e, "gen_prob": prob_g_e,
                "imp_pred": pred_i_e, "imp_prob": prob_i_e,
            })
            results_own.append({
                "true_kw": true_kw,
                "gen_pred": pred_g_o, "gen_prob": prob_g_o,
                "imp_pred": pred_i_o, "imp_prob": prob_i_o,
            })

    def metrics(results):
        n = len(results)
        gen_correct = sum(1 for r in results if r["gen_pred"] == r["true_kw"])
        imp_falseaccept = sum(1 for r in results if r["imp_pred"] == r["true_kw"])
        imp_rej = sum(1 for r in results if r["imp_pred"] == UNKNOWN_LABEL)
        mean_g = sum(r["gen_prob"] for r in results) / n
        mean_i = sum(r["imp_prob"] for r in results) / n
        return {
            "TAR": gen_correct / n,
            "FAR": imp_falseaccept / n,
            "REJ": imp_rej / n,
            "mean_prob_g": mean_g,
            "mean_prob_i": mean_i,
            "collapse_ratio": mean_i / max(mean_g, 1e-6),
        }

    enr = metrics(results_enrolled)
    own = metrics(results_own)

    print()
    print("=" * 78)
    print("FiLM v2 — train/eval distribution mismatch diagnostic")
    print("=" * 78)
    print(f"  {'Metric':>20s} | {'Enrolled (5-clip avg)':>22s} | {'Own audio (matches train)':>28s}")
    print(f"  {'-'*20} | {'-'*22} | {'-'*28}")
    for k in ["TAR", "FAR", "REJ"]:
        print(f"  {k:>20s} | {enr[k]*100:>21.2f}% | {own[k]*100:>27.2f}%")
    print(f"  {'mean P(true_kw) gen':>20s} | {enr['mean_prob_g']:>22.4f} | {own['mean_prob_g']:>28.4f}")
    print(f"  {'mean P(true_kw) imp':>20s} | {enr['mean_prob_i']:>22.4f} | {own['mean_prob_i']:>28.4f}")
    print(f"  {'collapse ratio':>20s} | {enr['collapse_ratio']:>22.4f} | {own['collapse_ratio']:>28.4f}")
    print()
    print(f"TAR delta (own − enrolled): {(own['TAR'] - enr['TAR'])*100:+.2f} pp")
    print(f"Training gen_acc at ep30  : 92.20 %  (reference)")

    summary = {
        "ckpt": Path(args.ckpt).name,
        "enrolled_eval": enr,
        "own_audio_eval": own,
        "TAR_delta_own_minus_enrolled_pp": round((own["TAR"] - enr["TAR"]) * 100, 2),
        "training_gen_acc_at_ep30": 0.922,
    }
    out_path = OUT_DIR / "diagnostic_v2.json"
    json.dump(summary, open(out_path, "w"), indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
