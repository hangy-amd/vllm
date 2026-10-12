"""Serialisation round-trip: a reloaded mapper transfers identically."""

import torch

from cross_model_kv.config import MapperConfig
from cross_model_kv.io import load_mapper, save_mapper
from cross_model_kv.mapper import RidgeKVMapper
from cross_model_kv.rope import RopeCodec
from tests.synthetic import make_synthetic


def test_save_load_roundtrip(tmp_path):
    cfg = MapperConfig(
        num_source_layers=6,
        num_target_layers=9,
        num_kv_heads=8,
        head_dim=128,
        top_k=3,
        source_model="synthetic-src",
        target_model="synthetic-tgt",
    )
    calib = make_synthetic(cfg, torch.arange(4096), seed=1)
    src_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.source_rope_theta)
    tgt_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta)
    src_k_c = src_codec.strip_at(calib.src_k_rope, calib.positions)
    tgt_k_c = tgt_codec.strip_at(calib.tgt_k_rope, calib.positions)
    mapper = RidgeKVMapper(cfg).fit(src_k_c, calib.src_v, tgt_k_c, calib.tgt_v)

    path = save_mapper(mapper, tmp_path / "qwen3_synth")
    assert path.exists()
    assert (tmp_path / "qwen3_synth.json").exists()

    reloaded = load_mapper(tmp_path / "qwen3_synth")
    assert reloaded.cfg.source_model == "synthetic-src"
    assert reloaded.cfg.top_k == 3

    k0, v0 = mapper.transfer(src_k_c, calib.src_v)
    k1, v1 = reloaded.transfer(src_k_c, calib.src_v)
    assert torch.allclose(k0, k1, atol=1e-5)
    assert torch.allclose(v0, v1, atol=1e-5)
    for t in range(cfg.num_target_layers):
        assert torch.equal(mapper.selected[t], reloaded.selected[t])
