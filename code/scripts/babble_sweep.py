"""Babble-robustness sweep — competing-speaker noise instead of MUSAN.

Tests how the KWS model handles cocktail-party scenarios where the "noise"
is intelligible speech from other speakers. Mirrors the existing MUSAN-noise
SNR sweep so the results are directly comparable.

Usage from code/:
    .venv/bin/python scripts/babble_sweep.py \\
        --ckpt "../results (1)/runs/p0/best.pt" \\
        --data_root ../data/speech_commands_v2 \\
        --n_voices 2 \\
        --out ../results/babble_sweep/p0_babble_n2.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from src.data import SpeechCommandsV2
from src.data.augment import _rms, SAMPLE_RATE
from src.train import KWSModel

CLIP_LEN = 16000


def load_clip(path: str) -> np.ndarray:
    wav, _ = sf.read(path)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    if len(wav) < CLIP_LEN:
        wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
    return wav[:CLIP_LEN]


def build_babble_pool(sc_root: Path, n_pool: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    all_clips = []
    for word_dir in sc_root.iterdir():
        if not word_dir.is_dir() or word_dir.name.startswith("_"):
            continue
        all_clips.extend(list(word_dir.glob("*.wav")))
    rng.shuffle(all_clips)
    return all_clips[:n_pool]


def make_babble(pool: list[Path], length: int, n_voices: int,
                rng: random.Random) -> np.ndarray:
    voices = rng.sample(pool, n_voices)
    babble = np.zeros(length, dtype=np.float32)
    for v in voices:
        wav = load_clip(str(v))
        if len(wav) < length:
            wav = np.tile(wav, (length // len(wav)) + 1)[:length]
        babble += wav
    babble = babble / max(n_voices, 1)
    return babble


def mix_at_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = _rms(wav)
    noise_rms = _rms(noise)
    if noise_rms < 1e-6 or sig_rms < 1e-6:
        return wav
    target = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target / noise_rms)
    out = wav + noise
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


@torch.no_grad()
def eval_at_snr(model, loader, snr_db, n_voices, babble_pool, device, seed):
    rng = random.Random(seed)
    total, correct = 0, 0
    for wav, label in loader:
        if snr_db < 900:
            mixed_batch = []
            for w in wav:
                w_np = w.numpy()
                babble = make_babble(babble_pool, len(w_np), n_voices, rng)
                out = mix_at_snr(w_np, babble, snr_db)
                mixed_batch.append(torch.from_numpy(out))
            wav = torch.stack(mixed_batch)
        wav = wav.to(device)
        label = label.to(device)
        logits = model(wav)
        pred = logits.argmax(dim=-1)
        total += int(label.size(0))
        correct += int((pred == label).sum().item())
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_voices", type=int, default=2,
                    help="number of competing speakers in the babble noise")
    ap.add_argument("--snrs", default="999,20,10,5,0,-5")
    ap.add_argument("--n_pool", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device      : {device}")
    print(f"ckpt        : {args.ckpt}")
    print(f"n_voices    : {args.n_voices}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    mcfg = cfg["model"]
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
    print(f"val items   : {len(val_ds)}")

    babble_pool = build_babble_pool(Path(args.data_root),
                                     n_pool=args.n_pool, seed=args.seed)
    print(f"babble pool : {len(babble_pool)} clips")

    snrs = [float(s) for s in args.snrs.split(",")]
    results = []
    for s in snrs:
        acc = eval_at_snr(model, loader, s, args.n_voices, babble_pool,
                          device, args.seed)
        label = "clean" if s >= 900 else f"{int(s)}dB"
        print(f"  babble SNR={label:>6s}  N={args.n_voices}  val_acc={acc:.4f}")
        results.append({"snr_db": s, "val_acc": acc, "n_voices": args.n_voices})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "val_acc", "n_voices"])
        w.writeheader()
        w.writerows(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
