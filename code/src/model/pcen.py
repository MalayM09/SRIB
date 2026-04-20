"""Learnable Per-Channel Energy Normalisation (Wang et al. 2017).

PCEN(E, M) = (E / (eps + M)^alpha + delta)^r - delta^r
with M[t] = (1 - s) * M[t-1] + s * E[t]  (first-order IIR smoother)

All of {alpha, delta, r, s} are per-channel, softplus-reparameterised so they
stay positive under any gradient update.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    # inverse of softplus: x = log(exp(y) - 1)
    return math.log(math.expm1(y))


class LearnablePCEN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        s_init: float = 0.025,
        alpha_init: float = 0.98,
        delta_init: float = 2.0,
        r_init: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.log_s = nn.Parameter(torch.full((num_channels,), _inv_softplus(s_init)))
        self.log_alpha = nn.Parameter(torch.full((num_channels,), _inv_softplus(alpha_init)))
        self.log_delta = nn.Parameter(torch.full((num_channels,), _inv_softplus(delta_init)))
        self.log_r = nn.Parameter(torch.full((num_channels,), _inv_softplus(r_init)))

    def _iir_smooth(self, E: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        # E: (B, F, T); s: (F,)
        # Build a list of per-timestep tensors and stack — avoids in-place writes
        # that would break autograd.
        T = E.size(-1)
        s_b = s.view(1, -1)
        one_minus_s = 1.0 - s_b
        prev = E[..., 0]
        out = [prev]
        for t in range(1, T):
            prev = one_minus_s * prev + s_b * E[..., t]
            out.append(prev)
        return torch.stack(out, dim=-1)

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        # E: (B, F, T) — magnitude mel-spectrogram (non-negative)
        assert E.dim() == 3 and E.size(1) == self.num_channels, (
            f"expected (B, {self.num_channels}, T), got {tuple(E.shape)}"
        )
        s = F.softplus(self.log_s)
        alpha = F.softplus(self.log_alpha).view(1, -1, 1)
        delta = F.softplus(self.log_delta).view(1, -1, 1)
        r = F.softplus(self.log_r).view(1, -1, 1)

        M = self._iir_smooth(E, s)
        out = (E / (self.eps + M).pow(alpha) + delta).pow(r) - delta.pow(r)
        return out
