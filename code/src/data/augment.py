"""Waveform- and spectrogram-domain augmentation primitives.

- WaveformAugment: RIR convolution + MUSAN mixing (applied inside the DataLoader worker).
- SpecAugment: time/freq masking, applied after mel extraction in the model.

Keeping this intentionally small — the Kaggle pilot adds codec-sim + SNR curriculum.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_LEN = 16000
SAMPLE_RATE = 16000


def _load_wavs(root: Path, patterns: tuple[str, ...] = ("*.wav",)) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.rglob(pat))
    return out


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


class RIRConvolver:
    """Convolve a waveform with a randomly sampled small-room RIR."""

    def __init__(self, rir_dir: str | Path, prob: float = 0.5, seed: int = 1337):
        self.paths = _load_wavs(Path(rir_dir))
        self.prob = prob
        self._rng = random.Random(seed)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if not self.paths or self._rng.random() > self.prob:
            return wav
        rir, sr = sf.read(str(self._rng.choice(self.paths)))
        if rir.ndim > 1:
            rir = rir[:, 0]
        if sr != SAMPLE_RATE:
            return wav
        rir = rir.astype(np.float32)
        peak = np.argmax(np.abs(rir))
        rir = rir[peak:peak + 8000]  # keep first 500 ms of RIR
        rir /= (np.max(np.abs(rir)) + 1e-9)
        conv = np.convolve(wav, rir, mode="full")[:len(wav)]
        # preserve loudness
        target_rms = _rms(wav)
        conv_rms = _rms(conv)
        if conv_rms > 1e-6:
            conv *= target_rms / conv_rms
        return conv.astype(np.float32)


class BabbleMixer:
    """Mix N random SC V2 keyword clips at uniformly-sampled SNR.
    Use SC V2 itself as the babble source — no extra dataset needed.
    Models cocktail-party 'background conversations' that MUSAN can't reproduce.
    """

    def __init__(
        self,
        sc_root: str | Path,
        prob: float = 0.4,
        n_voices_range: tuple[int, int] = (1, 2),
        snr_range: tuple[float, float] = (-5.0, 20.0),
        n_pool: int = 3000,
        seed: int = 1337,
    ):
        root = Path(sc_root)
        all_clips: List[Path] = []
        if root.is_dir():
            for word_dir in root.iterdir():
                if not word_dir.is_dir() or word_dir.name.startswith("_"):
                    continue
                all_clips.extend(list(word_dir.glob("*.wav")))
        rng_init = random.Random(seed)
        rng_init.shuffle(all_clips)
        self.paths = all_clips[:n_pool]
        self.prob = prob
        self.n_lo, self.n_hi = n_voices_range
        self.snr_lo, self.snr_hi = snr_range
        self._rng = random.Random(seed + 1)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if not self.paths or self._rng.random() > self.prob:
            return wav
        n_voices = self._rng.randint(self.n_lo, self.n_hi)
        babble = np.zeros_like(wav, dtype=np.float32)
        loaded = 0
        tries = 0
        while loaded < n_voices and tries < n_voices * 4:
            tries += 1
            v_path = self._rng.choice(self.paths)
            try:
                v, sr = sf.read(str(v_path))
                if v.ndim > 1:
                    v = v[:, 0]
                if sr != SAMPLE_RATE:
                    continue
                v = v.astype(np.float32)
                if len(v) < len(wav):
                    reps = (len(wav) // max(len(v), 1)) + 1
                    v = np.tile(v, reps)
                babble += v[:len(wav)]
                loaded += 1
            except Exception:
                continue
        if loaded == 0:
            return wav
        babble = babble / loaded
        snr_db = self._rng.uniform(self.snr_lo, self.snr_hi)
        sig_rms = _rms(wav)
        n_rms = _rms(babble)
        if n_rms < 1e-6 or sig_rms < 1e-6:
            return wav
        target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        babble = babble * (target_noise_rms / n_rms)
        return (wav + babble).astype(np.float32)


class MUSANMixer:
    """Mix a random MUSAN noise/music snippet at a uniformly-sampled SNR."""

    def __init__(
        self,
        musan_dir: str | Path,
        prob: float = 0.5,
        snr_range: tuple[float, float] = (0.0, 20.0),
        seed: int = 1337,
    ):
        root = Path(musan_dir)
        self.paths: List[Path] = _load_wavs(root / "noise") + _load_wavs(root / "music")
        self.prob = prob
        self.snr_lo, self.snr_hi = snr_range
        self._rng = random.Random(seed)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        if not self.paths or self._rng.random() > self.prob:
            return wav
        noise, sr = sf.read(str(self._rng.choice(self.paths)))
        if noise.ndim > 1:
            noise = noise[:, 0]
        if sr != SAMPLE_RATE:
            return wav
        noise = noise.astype(np.float32)
        if len(noise) < len(wav):
            reps = (len(wav) // len(noise)) + 1
            noise = np.tile(noise, reps)
        start = self._rng.randint(0, len(noise) - len(wav))
        noise = noise[start:start + len(wav)]

        snr_db = self._rng.uniform(self.snr_lo, self.snr_hi)
        sig_rms = _rms(wav)
        noise_rms = _rms(noise)
        if noise_rms < 1e-6:
            return wav
        target_noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        noise = noise * (target_noise_rms / noise_rms)
        return (wav + noise).astype(np.float32)


class WaveformAugment:
    """Compose optional RIR + MUSAN + Babble. Returns a torch.Tensor."""

    def __init__(
        self,
        rir: Optional[RIRConvolver] = None,
        musan: Optional[MUSANMixer] = None,
        babble: Optional[BabbleMixer] = None,
    ):
        self.rir = rir
        self.musan = musan
        self.babble = babble

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        arr = wav.numpy().copy()
        if self.rir is not None:
            arr = self.rir(arr)
        if self.musan is not None:
            arr = self.musan(arr)
        if self.babble is not None:
            arr = self.babble(arr)
        peak = float(np.max(np.abs(arr)) + 1e-9)
        if peak > 0.99:
            arr = arr * (0.99 / peak)
        return torch.from_numpy(arr.astype(np.float32))


class SpecAugment(nn.Module):
    """Time / frequency masking on a mel/PCEN spectrogram.

    Input  : (B, F, T)
    Output : (B, F, T)
    """

    def __init__(
        self,
        freq_mask: int = 10,
        time_mask: int = 20,
        num_freq_masks: int = 1,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.F = freq_mask
        self.T = time_mask
        self.nF = num_freq_masks
        self.nT = num_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        B, F_, T_ = spec.shape
        out = spec
        for _ in range(self.nF):
            f = int(torch.randint(0, self.F + 1, (1,)).item())
            if f > 0:
                f0 = int(torch.randint(0, max(F_ - f, 1), (1,)).item())
                out = out.clone()
                out[:, f0:f0 + f, :] = 0.0
        for _ in range(self.nT):
            t = int(torch.randint(0, self.T + 1, (1,)).item())
            if t > 0:
                t0 = int(torch.randint(0, max(T_ - t, 1), (1,)).item())
                out = out.clone()
                out[:, :, t0:t0 + t] = 0.0
        return out
