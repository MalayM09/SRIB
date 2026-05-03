"""MUSAN noise SNR sweep — mirror of babble_sweep.py but using MUSAN ambient noise."""
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
from src.data.augment import _rms, _load_wavs, SAMPLE_RATE
from src.train import KWSModel

CLIP_LEN = 16000


def mix_at_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(wav):
        reps = (len(wav) // max(len(noise), 1)) + 1
        noise = np.tile(noise, reps)
    noise = noise[:len(wav)].astype(np.float32)
    sig_rms = _rms(wav); n_rms = _rms(noise)
    if n_rms < 1e-6 or sig_rms < 1e-6:
        return wav
    target = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target / n_rms)
    out = wav + noise
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


@torch.no_grad()
def eval_at_snr(model, loader, snr_db, noise_paths, device, seed):
    rng = random.Random(seed)
    total, correct = 0, 0
    for wav, label in loader:
        if snr_db < 900:
            mixed = []
            for w in wav:
                w_np = w.numpy()
                noise = None
                for _ in range(5):
                    p = rng.choice(noise_paths)
                    try:
                        n, sr = sf.read(str(p))
                        if n.ndim > 1: n = n[:, 0]
                        if sr == SAMPLE_RATE:
                            noise = n.astype(np.float32); break
                    except Exception:
                        continue
                if noise is None:
                    mixed.append(w); continue
                if len(noise) > len(w_np):
                    start = rng.randint(0, len(noise) - len(w_np))
                    noise = noise[start:start + len(w_np)]
                mixed.append(torch.from_numpy(mix_at_snr(w_np, noise, snr_db)))
            wav = torch.stack(mixed)
        wav = wav.to(device); label = label.to(device)
        logits = model(wav)
        pred = logits.argmax(dim=-1)
        total += int(label.size(0))
        correct += int((pred == label).sum().item())
    return correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--musan_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--snrs", default="999,20,10,5,0,-5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"device   : {device}")
    print(f"ckpt     : {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]; mcfg = cfg["model"]
    model = KWSModel(
        n_mels=mcfg["n_mels"], num_classes=mcfg["num_classes"],
        use_pcen=mcfg.get("use_pcen", True), specaug=False,
        sample_rate=cfg["data"]["sample_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    val_ds = SpeechCommandsV2(args.data_root, split="val", seed=args.seed)
    loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"],
                        shuffle=False, num_workers=0)
    noise_paths = _load_wavs(Path(args.musan_dir) / "noise") + \
                  _load_wavs(Path(args.musan_dir) / "music")
    print(f"val items: {len(val_ds)}  noise files: {len(noise_paths)}")

    snrs = [float(s) for s in args.snrs.split(",")]
    results = []
    for s in snrs:
        acc = eval_at_snr(model, loader, s, noise_paths, device, args.seed)
        label = "clean" if s >= 900 else f"{int(s)}dB"
        print(f"  MUSAN SNR={label:>6s}  val_acc={acc:.4f}")
        results.append({"snr_db": s, "val_acc": acc})

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "val_acc"])
        w.writeheader(); w.writerows(results)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
