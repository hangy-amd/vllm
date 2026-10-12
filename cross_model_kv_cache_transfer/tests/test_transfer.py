"""Serving-time transfer math (``CrossModelKVConnector._map_sequence``).

The connector's injection path is split so the algorithm — strip source RoPE,
apply the fitted per-layer mapper, re-apply target RoPE — is a pure function of
the loaded source cache, testable without a live vLLM engine. We fabricate the
on-disk source cache the producer would write, load it through the real
``capture_store`` reader, and assert ``_map_sequence`` reproduces the mapper's
own batched transfer plus a clean target-RoPE round-trip.
"""

from __future__ import annotations

import torch

from cross_model_kv.capture_store import load_sequence
from cross_model_kv.config import MapperConfig
from cross_model_kv.mapper import RidgeKVMapper
from cross_model_kv.rope import RopeCodec
from cross_model_kv.transfer_connector import CrossModelKVConnector
from tests.synthetic import make_synthetic
from tests.test_capture_store import write_sequence


def _fitted_connector(cfg, calib):
    src_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.source_rope_theta)
    tgt_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta)
    src_k_c = src_codec.strip_at(calib.src_k_rope, calib.positions)
    tgt_k_c = tgt_codec.strip_at(calib.tgt_k_rope, calib.positions)
    mapper = RidgeKVMapper(cfg).fit(src_k_c, calib.src_v, tgt_k_c, calib.tgt_v)

    conn = CrossModelKVConnector.__new__(CrossModelKVConnector)
    conn._mapper = mapper
    conn._src_codec = src_codec
    conn._tgt_codec = tgt_codec
    return conn, mapper, src_k_c


def test_map_sequence_matches_batched_transfer_with_rope():
    cfg = MapperConfig(
        num_source_layers=6, num_target_layers=9, num_kv_heads=8, head_dim=128,
        top_k=3, source_model="src", target_model="tgt",
    )
    pos = torch.arange(4096)
    calib = make_synthetic(cfg, pos, seed=3)
    conn, mapper, src_k_c = _fitted_connector(cfg, calib)

    out_k, out_v = conn._map_sequence(calib.src_k_rope, calib.src_v, pos)
    assert len(out_k) == cfg.num_target_layers

    # Reference: the mapper's own batched transfer, then target RoPE on the keys.
    pred_k_c, pred_v = mapper.transfer(src_k_c, calib.src_v)
    tgt_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta)
    for t in range(cfg.num_target_layers):
        expect_k = tgt_codec.apply_at(pred_k_c[t], pos)
        assert torch.allclose(out_k[t], expect_k, atol=1e-4)
        assert torch.allclose(out_v[t], pred_v[t], atol=1e-4)


def test_map_sequence_reads_from_on_disk_capture(tmp_path):
    """End-to-end read path: disk shards -> load_sequence -> _map_sequence."""
    cfg = MapperConfig(
        num_source_layers=4, num_target_layers=6, num_kv_heads=8, head_dim=128,
        top_k=2, source_model="src", target_model="tgt",
    )
    pos = torch.arange(4096)
    calib = make_synthetic(cfg, pos, seed=5)
    conn, _, _ = _fitted_connector(cfg, calib)

    seq = tmp_path / "seqhash"
    write_sequence(seq, calib.src_k_rope, calib.src_v, pos.tolist())
    disk_k, disk_v = load_sequence(seq)
    assert torch.equal(disk_k, calib.src_k_rope)

    mem_k, mem_v = conn._map_sequence(calib.src_k_rope, calib.src_v, pos)
    on_k, on_v = conn._map_sequence(disk_k, disk_v, pos)
    for t in range(cfg.num_target_layers):
        assert torch.allclose(on_k[t], mem_k[t], atol=1e-5)
        assert torch.allclose(on_v[t], mem_v[t], atol=1e-5)
