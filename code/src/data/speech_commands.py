"""Speech Commands V2 loader with the canonical 10-word + unknown + silence split.

Uses the official `validation_list.txt` / `testing_list.txt` shipped inside the
tarball. Everything not listed there is training. `_silence_` clips are sampled
from the `_background_noise_/` directory; `_unknown_` samples come from the
remaining 25 commands.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


SC_V2_WORDS: List[str] = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
]
UNKNOWN_LABEL = 10
SILENCE_LABEL = 11
NUM_CLASSES = 12
SAMPLE_RATE = 16000
CLIP_LEN = 16000  # 1 second


def _read_list(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


class SpeechCommandsV2(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",   # {train, val, test}
        subset_size: int | None = None,
        silence_ratio: float = 0.1,
        unknown_ratio: float = 0.1,
        seed: int = 1337,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        rng = random.Random(seed)

        val_set = _read_list(self.root / "validation_list.txt")
        test_set = _read_list(self.root / "testing_list.txt")

        word_dirs = [p for p in self.root.iterdir()
                     if p.is_dir() and not p.name.startswith("_")]

        items: List[Tuple[str, int]] = []
        for d in word_dirs:
            word = d.name
            label = SC_V2_WORDS.index(word) if word in SC_V2_WORDS else UNKNOWN_LABEL
            for wav in d.glob("*.wav"):
                rel = f"{word}/{wav.name}"
                in_val = rel in val_set
                in_test = rel in test_set
                if split == "train" and not (in_val or in_test):
                    items.append((str(wav), label))
                elif split == "val" and in_val:
                    items.append((str(wav), label))
                elif split == "test" and in_test:
                    items.append((str(wav), label))

        # split known (0..9) vs unknown (=10) so we can rebalance `_unknown_`
        known = [x for x in items if x[1] != UNKNOWN_LABEL]
        unknown = [x for x in items if x[1] == UNKNOWN_LABEL]
        rng.shuffle(known)
        rng.shuffle(unknown)

        # cap unknowns so they don't dominate: ratio w.r.t. known
        target_unknown = int(len(known) * unknown_ratio / (1.0 - unknown_ratio - silence_ratio))
        unknown = unknown[:target_unknown]

        self.samples: List[Tuple[str, int]] = known + unknown

        # silence clips — sampled on the fly from background noise dir
        bg_dir = self.root / "_background_noise_"
        self.bg_waves: List[np.ndarray] = []
        if bg_dir.is_dir():
            for w in bg_dir.glob("*.wav"):
                data, sr = sf.read(str(w))
                if sr == SAMPLE_RATE:
                    self.bg_waves.append(data.astype(np.float32))

        target_silence = int(len(self.samples) * silence_ratio / (1.0 - silence_ratio))
        # encode silence as ("<SILENCE>", SILENCE_LABEL) sentinel paths
        for _ in range(target_silence):
            self.samples.append(("<SILENCE>", SILENCE_LABEL))

        rng.shuffle(self.samples)
        if subset_size is not None:
            self.samples = self.samples[:subset_size]

        self._rng = random.Random(seed + 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_silence(self) -> np.ndarray:
        if not self.bg_waves:
            return np.zeros(CLIP_LEN, dtype=np.float32)
        bg = self._rng.choice(self.bg_waves)
        if len(bg) < CLIP_LEN:
            return np.pad(bg, (0, CLIP_LEN - len(bg)))
        start = self._rng.randint(0, len(bg) - CLIP_LEN)
        clip = bg[start:start + CLIP_LEN]
        # scale silence down so it's not same energy as speech
        return (clip * self._rng.uniform(0.0, 0.3)).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        if path == "<SILENCE>":
            wav = self._load_silence()
        else:
            wav, sr = sf.read(path)
            assert sr == SAMPLE_RATE, f"expected 16 kHz, got {sr}"
            wav = wav.astype(np.float32)
            if len(wav) < CLIP_LEN:
                wav = np.pad(wav, (0, CLIP_LEN - len(wav)))
            elif len(wav) > CLIP_LEN:
                wav = wav[:CLIP_LEN]
        return torch.from_numpy(wav), label
