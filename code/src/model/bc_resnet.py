"""BC-ResNet-8 trunk (Kim et al., Interspeech 2021).

Follows the Qualcomm reference implementation layout:
- Inverted-residual blocks with expansion ratio 2:
    1x1 expand (c_in -> 2*c_out)  ->  3x1 DW freq (+ SubSpectralNorm)  ->
    freq-pool  ->  1x3 DW time (+ BN + SiLU)  ->  1x1 contract (2*c_out -> c_out)
- Stem: 5x5 conv, stride (2,1)  — halves freq axis.
- 4 stages: depths [2,2,4,4], channels [16,24,32,40], freq strides [2,2,2,1],
  time dilations [1,2,4,8] (one per stage).

Input : (B, 1, F=40, T=98)
Output: (B, 40, F', T)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class SubSpectralNorm(nn.Module):
    """Split the frequency axis into S groups and BN each group independently."""

    def __init__(self, num_channels: int, num_freq: int, num_groups: int = 5):
        super().__init__()
        if num_freq % num_groups != 0 or num_freq < num_groups:
            self.ssn = False
            self.bn = nn.BatchNorm2d(num_channels)
        else:
            self.ssn = True
            self.S = num_groups
            self.bn = nn.BatchNorm2d(num_channels * num_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ssn:
            return self.bn(x)
        B, C, F, T = x.shape
        x = x.view(B, C * self.S, F // self.S, T)
        x = self.bn(x)
        return x.view(B, C, F, T)


def _conv_bn_relu(c_in: int, c_out: int, ks=(1, 1), stride=(1, 1),
                  padding=(0, 0), groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=ks, stride=stride,
                  padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(c_out),
        nn.SiLU(inplace=True),
    )


class TransitionBlock(nn.Module):
    """First block in a stage — changes channel count and optionally freq stride."""

    def __init__(self, c_in: int, c_out: int, freq_out: int,
                 freq_stride: int = 1, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        mid = c_out * 2  # inverted-residual expansion

        # 1x1 expand (c_in -> mid)
        self.expand = _conv_bn_relu(c_in, mid, ks=(1, 1))

        # f2: freq depthwise 3x1 + SSN (optionally with stride)
        self.f2 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(3, 1), stride=(freq_stride, 1),
                      padding=(1, 0), groups=mid, bias=False),
            SubSpectralNorm(mid, freq_out, 5),
        )

        # f1: time depthwise 1x3 + BN + SiLU, applied after freq pooling
        self.f1 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(1, 3),
                      padding=(0, dilation), dilation=(1, dilation),
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # 1x1 contract (mid -> c_out)
        self.project = nn.Sequential(
            nn.Conv2d(mid, c_out, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(c_out),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)               # (B, mid, F, T)
        x = self.f2(x)                   # (B, mid, F', T)
        pooled = x.mean(dim=2, keepdim=True)     # (B, mid, 1, T)
        f1_out = self.f1(pooled)                  # (B, mid, 1, T)
        x = x + f1_out                            # broadcast over F
        return self.project(x)


class NormalBlock(nn.Module):
    """Residual block — no channel / stride change."""

    def __init__(self, channels: int, freq: int,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        mid = channels * 2

        self.expand = _conv_bn_relu(channels, mid, ks=(1, 1))
        self.f2 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(3, 1), padding=(1, 0),
                      groups=mid, bias=False),
            SubSpectralNorm(mid, freq, 5),
        )
        self.f1 = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=(1, 3),
                      padding=(0, dilation), dilation=(1, dilation),
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid, channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.expand(x)
        y = self.f2(y)
        pooled = y.mean(dim=2, keepdim=True)
        y = y + self.f1(pooled)
        y = self.project(y)
        return identity + y


class BCResNet8(nn.Module):
    """BC-ResNet-8 (tau=8). Returns a feature map; pooling happens in the head.

    Channels can be overridden per-instance via the `channels` constructor arg
    (e.g. [60, 90, 120, 150] for the BC-ResNet-9 width target — ~900K trunk).
    """

    DEPTHS: List[int] = [2, 2, 4, 4]
    # Default τ=8 width — ~321K trunk; matches the paper's parameter budget
    CHANNELS: List[int] = [40, 60, 80, 100]
    FREQ_STRIDES: List[int] = [2, 2, 2, 1]
    DILATIONS: List[int] = [1, 2, 4, 8]

    def __init__(self, n_mels: int = 40, dropout: float = 0.1,
                 channels: List[int] | None = None):
        super().__init__()
        self.n_mels = n_mels
        self._channels = channels if channels is not None else self.CHANNELS

        self.stem = nn.Sequential(
            nn.Conv2d(1, self._channels[0], kernel_size=(5, 5),
                      stride=(2, 1), padding=(2, 2), bias=False),
            nn.BatchNorm2d(self._channels[0]),
            nn.SiLU(inplace=True),
        )

        blocks: List[nn.Module] = []
        c_prev = self._channels[0]
        freq = n_mels // 2  # after stem stride

        for stage, (depth, c_out, fs, dil) in enumerate(
            zip(self.DEPTHS, self._channels, self.FREQ_STRIDES, self.DILATIONS)
        ):
            for b in range(depth):
                is_first = (b == 0)
                if is_first:
                    freq_after = freq // fs if fs > 1 else freq
                    blocks.append(TransitionBlock(
                        c_in=c_prev, c_out=c_out, freq_out=freq_after,
                        freq_stride=fs, dilation=dil, dropout=dropout,
                    ))
                    c_prev = c_out
                    freq = freq_after
                else:
                    blocks.append(NormalBlock(
                        channels=c_out, freq=freq, dilation=dil, dropout=dropout,
                    ))

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = c_prev
        self.out_freq = freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
