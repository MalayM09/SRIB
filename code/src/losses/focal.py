"""Focal cross-entropy (Lin et al. 2017) — helps with `_silence_` / `_unknown_` imbalance."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()
