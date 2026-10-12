"""Synthetic paired caches with genuine cross-layer linear structure.

Used to exercise the mapper end-to-end without running real models: draw random
source content, propagate it through a hidden ground-truth cross-layer linear
mix, and apply RoPE to the keys. The mapper must recover the relationship from
the *stored* caches alone (keys carry RoPE, values do not).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cross_model_kv.config import MapperConfig
from cross_model_kv.rope import RopeCodec

Tensor = torch.Tensor


@dataclass
class SyntheticPair:
    cfg: MapperConfig
    src_k_rope: Tensor      # [L_s, T, H, D] stored source keys (RoPE applied)
    src_v: Tensor           # [L_s, T, H, D]
    tgt_k_rope: Tensor      # [L_t, T, H, D] stored target keys (RoPE applied)
    tgt_v: Tensor           # [L_t, T, H, D]
    positions: Tensor       # [T]
    true_sources: list[Tensor]   # per target layer: source indices actually used


def make_synthetic(
    cfg: MapperConfig,
    positions: Tensor,
    *,
    true_sources_per_target: int = 3,
    noise_std: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
) -> SyntheticPair:
    gen = torch.Generator(device=device).manual_seed(seed)
    n_kv, d_h = cfg.num_kv_heads, cfg.head_dim
    ls, lt = cfg.num_source_layers, cfg.num_target_layers
    num_tokens = positions.shape[0]
    width = n_kv * d_h
    k = true_sources_per_target
    scale = (k * width) ** -0.5

    sources = [
        torch.randperm(ls, generator=gen, device=device)[:k].sort().values
        for _ in range(lt)
    ]
    key_mix = [
        torch.randn(k * width, width, generator=gen, device=device) * scale
        for _ in range(lt)
    ]
    val_mix = [
        torch.randn(k * width, width, generator=gen, device=device) * scale
        for _ in range(lt)
    ]

    shape = (num_tokens, n_kv, d_h)
    src_k_c = torch.randn(ls, *shape, generator=gen, device=device)
    src_v = torch.randn(ls, *shape, generator=gen, device=device)

    # The mixing matmuls dominate generation; run them on an accelerator when
    # present (CPU BLAS is a slow reference gemm on LAPACK-less ROCm builds).
    mm_dev = "cuda" if (device == "cpu" and torch.cuda.is_available()) else device

    def mix(src: Tensor, mixes: list[Tensor]) -> Tensor:
        out = torch.empty(lt, *shape, device=device)
        for layer in range(lt):
            x = torch.cat(
                [src[s].reshape(num_tokens, -1) for s in sources[layer]], dim=-1
            ).to(mm_dev)
            noise = torch.randn(num_tokens, width, generator=gen, device=device)
            mixed = (x @ mixes[layer].to(mm_dev)).to(device)
            out[layer] = (mixed + noise_std * noise).reshape(shape)
        return out

    tgt_k_c = mix(src_k_c, key_mix)
    tgt_v = mix(src_v, val_mix)

    src_codec = RopeCodec(d_h, cfg.rotary_dim, cfg.source_rope_theta, device)
    tgt_codec = RopeCodec(d_h, cfg.rotary_dim, cfg.target_rope_theta, device)
    return SyntheticPair(
        cfg=cfg,
        src_k_rope=src_codec.apply_at(src_k_c, positions),
        src_v=src_v,
        tgt_k_rope=tgt_codec.apply_at(tgt_k_c, positions),
        tgt_v=tgt_v,
        positions=positions,
        true_sources=sources,
    )
