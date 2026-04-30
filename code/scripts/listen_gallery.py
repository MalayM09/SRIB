"""Render a listenable gallery of raw + augmented audio.

Picks a few SC V2 keyword clips, a few RIRs, and (if available) a few MUSAN
noise clips, and writes side-by-side WAVs to data/listen/ so you can play
them in Finder / QuickLook and confirm the augmentation is meaningful.

If MUSAN is not present locally, a pink-noise fallback is used so you can
at least hear what SNR mixing does to speech. Download the real MUSAN with
  bash code/scripts/download_musan_small.sh
to hear the actual distribution used for training.

Usage:
  cd code && python scripts/listen_gallery.py
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SC_ROOT    = ROOT / "data" / "speech_commands_v2"
RIR_ROOT   = ROOT / "data" / "rirs_small"
MUSAN_ROOT = ROOT / "data" / "musan_small"
OUT        = ROOT / "data" / "listen"

KEYWORDS   = ["yes", "no", "stop", "go", "up", "down"]
SNRS_DB    = [20, 10, 5, 0, -5]
SAMPLE_RATE = 16000
CLIP_LEN    = SAMPLE_RATE


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def pad_or_crop(wav: np.ndarray, length: int = CLIP_LEN) -> np.ndarray:
    if len(wav) < length:
        return np.pad(wav, (0, length - len(wav)))
    return wav[:length]


def apply_rir(wav: np.ndarray, rir_path: Path) -> np.ndarray:
    rir, sr = sf.read(str(rir_path))
    if rir.ndim > 1:
        rir = rir[:, 0]
    if sr != SAMPLE_RATE:
        return wav
    rir = rir.astype(np.float32)
    peak = int(np.argmax(np.abs(rir)))
    rir = rir[peak:peak + 8000]
    rir = rir / (np.max(np.abs(rir)) + 1e-9)
    conv = np.convolve(wav, rir, mode="full")[:len(wav)]
    t_rms, c_rms = _rms(wav), _rms(conv)
    if c_rms > 1e-6:
        conv = conv * (t_rms / c_rms)
    return conv.astype(np.float32)


def mix_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(wav):
        reps = (len(wav) // len(noise)) + 1
        noise = np.tile(noise, reps)
    noise = noise[:len(wav)].astype(np.float32)
    sig_rms, n_rms = _rms(wav), _rms(noise)
    if n_rms < 1e-6 or sig_rms < 1e-6:
        return wav
    target = sig_rms / (10 ** (snr_db / 20.0))
    noise = noise * (target / n_rms)
    out = wav + noise
    peak = float(np.max(np.abs(out)) + 1e-9)
    if peak > 0.99:
        out = out * (0.99 / peak)
    return out.astype(np.float32)


def pink_noise(length: int, seed: int = 0) -> np.ndarray:
    """Fallback noise when MUSAN isn't downloaded. ~1/f spectrum."""
    rng = np.random.default_rng(abs(int(seed)))
    white = rng.standard_normal(length).astype(np.float32)
    freqs = np.fft.rfftfreq(length, d=1.0 / SAMPLE_RATE)
    freqs[0] = 1.0
    scaling = 1.0 / np.sqrt(freqs)
    spec = np.fft.rfft(white) * scaling
    pink = np.fft.irfft(spec, n=length).astype(np.float32)
    pink = pink / (np.max(np.abs(pink)) + 1e-9) * 0.3
    return pink


def find_rirs() -> list[Path]:
    cand = list((RIR_ROOT / "RIRS_NOISES" / "simulated_rirs").rglob("*.wav"))
    return cand or list(RIR_ROOT.rglob("*.wav"))


def find_musan_noises() -> list[Path]:
    if not MUSAN_ROOT.exists():
        return []
    return list((MUSAN_ROOT / "noise").rglob("*.wav")) or \
           list((MUSAN_ROOT / "music").rglob("*.wav")) or \
           list(MUSAN_ROOT.rglob("*.wav"))


def pick_keyword_clip(word: str) -> Path | None:
    d = SC_ROOT / word
    wavs = sorted(d.glob("*.wav"))
    return wavs[0] if wavs else None


def write_wav(path: Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    rirs = find_rirs()
    noises = find_musan_noises()
    use_pink = not noises

    print(f"SC V2 root : {SC_ROOT}  exists={SC_ROOT.exists()}")
    print(f"RIRs found : {len(rirs)}")
    print(f"MUSAN files: {len(noises)}  (fallback to pink noise: {use_pink})")
    print()

    # ------------- 1) Reference clips so you know what each component sounds like
    print("=== Reference clips ===")
    for i, rir_path in enumerate(rirs[:2]):
        rir, sr = sf.read(str(rir_path))
        if rir.ndim > 1:
            rir = rir[:, 0]
        write_wav(OUT / f"0_ref_rir_{i}.wav", rir.astype(np.float32), sr)

    if noises:
        for i, n_path in enumerate(noises[:3]):
            n, sr = sf.read(str(n_path))
            if n.ndim > 1:
                n = n[:, 0]
            # clip to 3 s for quick preview
            write_wav(OUT / f"0_ref_musan_noise_{i}.wav",
                      n[:sr * 3].astype(np.float32), sr)
    else:
        write_wav(OUT / "0_ref_pink_noise.wav",
                  pink_noise(SAMPLE_RATE * 3, seed=0))

    # ------------- 2) Per-keyword gallery
    print("\n=== Augmentation gallery ===")
    for word in KEYWORDS[:3]:
        clip_path = pick_keyword_clip(word)
        if clip_path is None:
            print(f"  skipping '{word}' — no clips found")
            continue
        wav, sr = sf.read(str(clip_path))
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = pad_or_crop(wav.astype(np.float32))
        if sr != SAMPLE_RATE:
            print(f"  WARN: '{word}' sr={sr}, expected {SAMPLE_RATE}")

        sub = OUT / word
        # a) raw clean
        write_wav(sub / f"01_clean.wav", wav, sr)

        # b) RIR only (reverb)
        if rirs:
            rir_path = rng.choice(rirs)
            write_wav(sub / f"02_rir_only.wav", apply_rir(wav, rir_path), sr)

        # c) Noise-only mix at each SNR
        for snr in SNRS_DB:
            if noises:
                np_path = rng.choice(noises)
                n, nsr = sf.read(str(np_path))
                if n.ndim > 1: n = n[:, 0]
                if nsr != sr:
                    continue
                noise = n.astype(np.float32)
            else:
                noise = pink_noise(len(wav) * 2, seed=snr)
            mixed = mix_snr(wav, noise, snr)
            write_wav(sub / f"03_noise_{snr:+03d}dB.wav", mixed, sr)

        # d) Full pipeline: RIR + noise at -5 dB (worst-case training sample)
        if rirs:
            rir_path = rng.choice(rirs)
            reverbed = apply_rir(wav, rir_path)
            if noises:
                np_path = rng.choice(noises)
                n, nsr = sf.read(str(np_path))
                if n.ndim > 1: n = n[:, 0]
                noise = n.astype(np.float32) if nsr == sr else pink_noise(len(wav) * 2)
            else:
                noise = pink_noise(len(wav) * 2, seed=99)
            combo = mix_snr(reverbed, noise, -5)
            write_wav(sub / f"04_rir_plus_noise_-5dB.wav", combo, sr)

    print()
    print("=" * 60)
    print(f"Done. Open in Finder:  {OUT}")
    print("Tip: select all WAVs and press Space for QuickLook preview.")
    print("     Or double-click to play in the default player.")
    if use_pink:
        print()
        print("NOTE: MUSAN wasn't found at data/musan_small/. Pink noise was")
        print("used as a fallback. To hear the real MUSAN distribution:")
        print("    bash code/scripts/download_musan_small.sh   (~1 GB final)")


if __name__ == "__main__":
    main()
