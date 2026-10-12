"""RoPE codec: strip is the exact inverse of apply, and matches NEOX semantics."""

import pytest
import torch

from cross_model_kv.rope import RopeCodec, _rotate_half


@pytest.mark.parametrize("rotary_dim", [128, 64])
@pytest.mark.parametrize("theta", [10_000.0, 1_000_000.0])
def test_strip_inverts_apply(rotary_dim, theta):
    torch.manual_seed(0)
    head_dim = 128
    codec = RopeCodec(head_dim, rotary_dim=rotary_dim, theta=theta)
    positions = torch.arange(0, 4096)
    x = torch.randn(len(positions), 8, head_dim)  # [T, H, D]
    round_trip = codec.strip_at(codec.apply_at(x, positions), positions)
    assert torch.allclose(round_trip, x, atol=1e-4, rtol=1e-4)


def test_apply_matches_neox_pairwise_formula():
    """apply == vLLM NEOX kernel:  o1 = x1 c - x2 s ; o2 = x2 c + x1 s."""
    torch.manual_seed(1)
    head_dim = 64
    codec = RopeCodec(head_dim, theta=10_000.0)
    positions = torch.arange(0, 17)
    x = torch.randn(len(positions), 4, head_dim)
    cos, sin = codec.cos_sin(positions)          # [T, 1, D] doubled
    got = codec.apply(x, cos, sin)

    half = head_dim // 2
    c = cos[..., :half]
    s = sin[..., :half]
    x1, x2 = x[..., :half], x[..., half:]
    expect = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)
    assert torch.allclose(got, expect, atol=1e-5)


def test_batched_layer_axis_broadcasts():
    """cos/sin broadcast over a leading layer axis: [L, T, H, D]."""
    torch.manual_seed(2)
    codec = RopeCodec(128, theta=1_000_000.0)
    positions = torch.arange(0, 32)
    x = torch.randn(5, len(positions), 8, 128)   # [L, T, H, D]
    round_trip = codec.strip_at(codec.apply_at(x, positions), positions)
    assert torch.allclose(round_trip, x, atol=1e-4)


def test_rotate_half():
    x = torch.arange(8).float().reshape(1, 1, 8)
    got = _rotate_half(x)
    expect = torch.tensor([-4.0, -5, -6, -7, 0, 1, 2, 3]).reshape(1, 1, 8)
    assert torch.equal(got, expect)
