"""Task heads. Local pilot: KWSHead is real; SVHead is a stub (Kaggle fills it)."""
from __future__ import annotations

import torch
import torch.nn as nn


class KWSHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 12, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, C, F', T)
        x = self.pool(feats).flatten(1)  # (B, C)
        return self.fc(self.drop(x))


class SVHead(nn.Module):
    """Speaker-verification stub — emits a zero 192-d embedding locally.

    Real implementation (MQMHA pool + AAM-Softmax) lands on Kaggle.
    """

    def __init__(self, in_channels: int, embed_dim: int = 192):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.size(0)
        return feats.new_zeros((B, self.embed_dim))
