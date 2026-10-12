"""Mapper recovery on synthetic structured caches (the paper's pipeline)."""

import torch

from cross_model_kv.config import MapperConfig
from cross_model_kv.diagnostics import attention_cosine
from cross_model_kv.mapper import RidgeKVMapper
from cross_model_kv.ridge import r2_score
from cross_model_kv.rope import RopeCodec
from tests.synthetic import make_synthetic


def _cfg() -> MapperConfig:
    return MapperConfig(
        num_source_layers=8,
        num_target_layers=12,
        num_kv_heads=8,
        head_dim=128,
        source_rope_theta=10_000.0,
        target_rope_theta=1_000_000.0,
        top_k=4,
        ridge_lambda=1e-2,
    )


def _fit_on(cfg, calib):
    """Fit a mapper in content space from a stored (RoPE'd) calibration pair."""
    src_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.source_rope_theta)
    tgt_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta)
    src_k_c = src_codec.strip_at(calib.src_k_rope, calib.positions)
    tgt_k_c = tgt_codec.strip_at(calib.tgt_k_rope, calib.positions)
    mapper = RidgeKVMapper(cfg).fit(src_k_c, calib.src_v, tgt_k_c, calib.tgt_v)
    return mapper, src_codec, tgt_codec


def test_selection_recovers_true_sources():
    cfg = _cfg()
    calib = make_synthetic(cfg, torch.arange(6144), true_sources_per_target=3, seed=0)
    mapper, _, _ = _fit_on(cfg, calib)
    # The k=4 selection must contain all 3 truly-informative source layers.
    for t in range(cfg.num_target_layers):
        picked = set(mapper.selected[t].tolist())
        truth = set(calib.true_sources[t].tolist())
        assert truth.issubset(picked), f"target {t}: {truth} not in {picked}"


def test_transfer_quality_on_unseen_positions():
    """Content-space fit generalises to later, unseen positions."""
    cfg = _cfg()
    calib = make_synthetic(cfg, torch.arange(6144), seed=0)
    # Same hidden relationship, evaluated at strictly later positions.
    ev = make_synthetic(cfg, torch.arange(6144, 6144 + 1024), seed=0)

    mapper, src_codec, tgt_codec = _fit_on(cfg, calib)
    src_k_ev_c = src_codec.strip_at(ev.src_k_rope, ev.positions)
    tgt_k_ev_c = tgt_codec.strip_at(ev.tgt_k_rope, ev.positions)

    pred_k_c, pred_v = mapper.transfer(src_k_ev_c, ev.src_v)
    assert r2_score(pred_k_c, tgt_k_ev_c) > 0.9
    assert r2_score(pred_v, ev.tgt_v) > 0.9

    # Attention-output cosine with random RoPE-encoded queries stays high.
    q_c = torch.randn(ev.positions.shape[0], cfg.num_kv_heads, cfg.head_dim)
    q_rope = tgt_codec.apply_at(q_c, ev.positions)
    pred_k_rope = tgt_codec.apply_at(pred_k_c, ev.positions)
    cos = sum(
        attention_cosine(
            q_rope, ev.tgt_k_rope[l], ev.tgt_v[l], pred_k_rope[l], pred_v[l]
        )
        for l in range(cfg.num_target_layers)
    ) / cfg.num_target_layers
    assert cos > 0.95


def test_content_space_beats_rope_coupled_at_unseen_positions():
    """The Sec 3.3 motivation: coupling the fit to RoPE degrades out-of-range."""
    cfg = _cfg()
    calib = make_synthetic(cfg, torch.arange(6144), seed=0)
    ev = make_synthetic(cfg, torch.arange(6144, 6144 + 1024), seed=0)
    mapper, src_codec, tgt_codec = _fit_on(cfg, calib)

    # Content-space keys R^2 (unseen positions).
    src_k_ev_c = src_codec.strip_at(ev.src_k_rope, ev.positions)
    tgt_k_ev_c = tgt_codec.strip_at(ev.tgt_k_rope, ev.positions)
    pred_k_c, _ = mapper.transfer(src_k_ev_c, ev.src_v)
    r2_content = r2_score(pred_k_c, tgt_k_ev_c)

    # RoPE-coupled: reuse the selection but fit directly on stored (rotated) keys.
    coupled = RidgeKVMapper(cfg)
    coupled.selected = mapper.selected
    coupled.key_maps = coupled._fit_cache(calib.src_k_rope, calib.tgt_k_rope)
    pred_k_coupled = coupled._transfer_cache(ev.src_k_rope, coupled.key_maps)
    r2_coupled = r2_score(pred_k_coupled, ev.tgt_k_rope)

    assert r2_content > r2_coupled + 0.2


def test_transfer_layer_matches_full_transfer():
    cfg = _cfg()
    calib = make_synthetic(cfg, torch.arange(6144), seed=0)
    mapper, src_codec, _ = _fit_on(cfg, calib)
    src_k_c = src_codec.strip_at(calib.src_k_rope, calib.positions)
    full_k, full_v = mapper.transfer(src_k_c, calib.src_v)
    for t in (0, 5, cfg.num_target_layers - 1):
        k, v = mapper.transfer_layer(src_k_c, calib.src_v, t)
        assert torch.allclose(k, full_k[t], atol=1e-4)
        assert torch.allclose(v, full_v[t], atol=1e-4)


def test_under_determined_raises():
    cfg = _cfg()
    import pytest

    calib = make_synthetic(cfg, torch.arange(64), seed=0)  # N < feature_dim
    with pytest.raises(ValueError, match="under-determined"):
        _fit_on(cfg, calib)
