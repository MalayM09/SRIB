"""Unit-level sanity checks — must all pass before a training run is attempted."""
from __future__ import annotations

import torch

from src.model import BCResNet8, KWSHead, LearnablePCEN, SVHead


def test_pcen_shape():
    pcen = LearnablePCEN(num_channels=40)
    x = torch.rand(2, 40, 98).abs() + 1e-3  # mel magnitudes
    y = pcen(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_pcen_gradients():
    pcen = LearnablePCEN(num_channels=40)
    x = torch.rand(2, 40, 98).abs() + 1e-3
    y = pcen(x).sum()
    y.backward()
    for name, p in pcen.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"


def test_bcresnet_forward_and_param_count():
    m = BCResNet8(n_mels=40)
    x = torch.randn(2, 1, 40, 98)
    y = m(x)
    assert y.dim() == 4
    assert y.shape[0] == 2
    assert y.shape[1] == m.out_channels
    n = m.num_parameters()
    # Sanity band around the paper's BC-ResNet-8 (tau=8) ~321K target.
    # Our inverted-residual block is ~5x cheaper per block, so we widen
    # channels to [40, 60, 80, 100] to match the paper's param budget.
    assert 250_000 <= n <= 400_000, f"unexpected trunk param count: {n}"


def test_kws_head_end_to_end():
    m = BCResNet8(n_mels=40)
    h = KWSHead(m.out_channels, num_classes=12)
    x = torch.randn(3, 1, 40, 98)
    logits = h(m(x))
    assert logits.shape == (3, 12)


def test_sv_head_stub_shape():
    m = BCResNet8(n_mels=40)
    h = SVHead(m.out_channels, embed_dim=192)
    x = torch.randn(4, 1, 40, 98)
    emb = h(m(x))
    assert emb.shape == (4, 192)


def test_total_param_budget_under_cap():
    """Sanity: trunk + KWS head well under the 3M competition cap."""
    m = BCResNet8(n_mels=40)
    h = KWSHead(m.out_channels, num_classes=12)
    total = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in h.parameters())
    assert total < 3_000_000, f"over cap: {total:,}"
