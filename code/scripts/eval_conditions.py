"""Evaluate a trained KWS checkpoint under each gallery condition.

Conditions mirror the listen_gallery.py output:
  clean, rir_only, musan_+20dB, musan_+10dB, musan_+5dB, musan_+0dB,
  musan_-5dB, rir_plus_musan_-5dB

Writes a CSV per checkpoint. Deterministic per (seed, condition) so two
different checkpoints see the same noise/RIR samples at each condition.

Usage:
    cd code && .venv/bin/python scripts/eval_conditions.py \\
        --ckpt "../results (1)/runs/p1_pcen_on/best.pt" \\
        --data_root ../data/speech_commands_v2 \\
        --rir_dir ../data/rirs_small \\
        --musan_dir ../data/musan_small \\
        --out conditions_p1_pcen_on.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from src.data import SpeechCommandsV2
from src.data.augment import _load_wavs, _rms, SAMPLE_RATE
from src.train import KWSModel


def build_noise_paths(musan_dir: Path):
    return _load_wavs(musan_dir / "noise") + _load_wavs(musan_dir / "music")


def build_rir_paths(rir_dir: Path):
    cand = list((rir_dir / "RIRS_NOISES" / "simulated_rirs").rglob("*.wav"))
    return cand or list(rir_dir.rglob("*.wav"))


def apply_rir(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:peak + 8000]
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    conv = np.convolve(wav, rir, mode="full")[:len(wav)]
    t_rms, c_rms = _rms(wav), _rms(conv)
    if c_rms > 1e-6:
        conv = conv * (t_rms / c_rms)
    return conv.astype(np.float32)


def mix_at_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(wav):
        reps = (len(wav) // len(noise)) + 1
        noise = np.tile(noise, reps)
    noise = noise[:len(wav)].astype(np.float32)
    sig_rms, n_rms = _rms(wav), _rms(noise)
    if n_rms < 1e-6 or sig_rms < 1e-6:
        return wav
    target = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target / n_rms)
    out = (wav + noise).astype(np.float32)
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out


def load_one_rir(paths, rng, target_sr=SAMPLE_RATE):
    for _ in range(5):
        p = rng.choice(paths)
        try:
            rir, sr = sf.read(str(p))
            if rir.ndim > 1:
                rir = rir[:, 0]
            if sr == target_sr:
                return rir.astype(np.float32)
        except Exception:
            pass
    return None


def load_one_noise(paths, rng, target_sr=SAMPLE_RATE):
    for _ in range(5):
        p = rng.choice(paths)
        try:
            n, sr = sf.read(str(p))
            if n.ndim > 1:
                n = n[:, 0]
            if sr == target_sr:
                return n.astype(np.float32)
        except Exception:
            pass
    return None


@torch.no_grad()
def eval_condition(model, loader, condition, rir_paths, noise_paths, device, seed):
    """Conditions: 'clean', 'rir_only', 'musan_<snr>', 'rir_plus_musan_<snr>'."""
    rng = random.Random(seed)
    total, correct = 0, 0
    for wav_batch, label_batch in loader:
        if condition != "clean":
            mixed = []
            for w in wav_batch:
                w_np = w.numpy()
                out = w_np
                if condition.startswith("rir") and rir_paths:
                    rir = load_one_rir(rir_paths, rng)
                    if rir is not None:
                        out = apply_rir(out, rir)
                if "musan" in condition and noise_paths:
                    snr_part = condition.rsplit("_", 1)[1].replace("dB", "")
                    snr_db = float(snr_part)
                    noise_raw = load_one_noise(noise_paths, rng)
                    if noise_raw is not None:
                        if len(noise_raw) > len(out):
                            start = rng.randint(0, len(noise_raw) - len(out))
                            noise_raw = noise_raw[start:start + len(out)]
                        out = mix_at_snr(out, noise_raw, snr_db)
                mixed.append(torch.from_numpy(out))
            wav_batch = torch.stack(mixed)
        wav_batch = wav_batch.to(device, non_blocking=True)
        label_batch = label_batch.to(device, non_blocking=True)
        logits = model(wav_batch)
        total += label_batch.size(0)
        correct += (logits.argmax(-1) == label_batch).sum().item()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True, help="path to speech_commands_v2/")
    ap.add_argument("--rir_dir", required=True)
    ap.add_argument("--musan_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None, help="cpu|mps|cuda (auto if omitted)")
    ap.add_argument("--conditions", default=None,
                    help="comma list to subset, e.g. 'clean,rir_only,musan_20dB'")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else
                          "mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"device: {device}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]; mcfg = cfg["model"]

    model = KWSModel(
        n_mels=mcfg["n_mels"],
        num_classes=mcfg["num_classes"],
        use_pcen=mcfg.get("use_pcen", True),
        specaug=False,
        sample_rate=cfg["data"]["sample_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_ds = SpeechCommandsV2(args.data_root, split="val", seed=args.seed)
    loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                        shuffle=False, num_workers=0)
    rir_paths = build_rir_paths(Path(args.rir_dir))
    noise_paths = build_noise_paths(Path(args.musan_dir))
    print(f"ckpt={args.ckpt} | val={len(val_ds)} | rir={len(rir_paths)} | noise={len(noise_paths)}")

    conditions = [
        ("clean",                   "clean"),
        ("rir_only",                "02_rir_only"),
        ("musan_20dB",              "03_noise_+20dB"),
        ("musan_10dB",              "03_noise_+10dB"),
        ("musan_5dB",               "03_noise_+05dB"),
        ("musan_0dB",               "03_noise_+00dB"),
        ("musan_-5dB",              "03_noise_-05dB"),
        ("rir_plus_musan_-5dB",     "04_rir_plus_noise_-5dB"),
    ]
    if args.conditions:
        wanted = set(args.conditions.split(","))
        conditions = [c for c in conditions if c[0] in wanted]

    results = []
    for cond, gallery_file in conditions:
        acc = eval_condition(model, loader, cond, rir_paths, noise_paths, device, args.seed)
        print(f"  {cond:>22s}  ({gallery_file:>30s})  val_acc={acc:.4f}")
        results.append({"condition": cond, "gallery_file": gallery_file, "val_acc": acc})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "gallery_file", "val_acc"])
        w.writeheader(); w.writerows(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
