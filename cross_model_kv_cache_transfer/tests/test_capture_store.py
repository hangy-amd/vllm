"""On-disk capture format: round-trip, layer ordering, hash + alignment rules.

Exercises the vLLM-free ``capture_store`` module that keeps the producer
(``KVCaptureConnector``) and consumer (``CrossModelKVConnector``) in lock-step,
without needing a live engine: we fabricate the exact files the producer writes
and assert the reader recovers them, in decoder order, byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from cross_model_kv.capture_store import (
    capture_align,
    hash_tokens,
    layer_index,
    load_manifest,
    load_sequence,
    num_layers,
    sequence_dirs,
    serve_matched,
)


def write_sequence(
    seq_dir: Path, k: torch.Tensor, v: torch.Tensor, token_ids: list[int]
) -> None:
    """Write ``[L, N, H, D]`` K/V as the per-layer ``{"kv": [N, 2, H, D]}`` shards
    the capture connector produces (shuffled filesystem order on purpose)."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    order = list(range(k.shape[0]))
    for i in reversed(order):  # write high layers first: filename sort must win
        kv = torch.stack((k[i], v[i]), dim=1)  # [N, 2, H, D]
        save_file(
            {"kv": kv.contiguous()},
            str(seq_dir / f"model.layers.{i}.self_attn.attn.safetensors"),
        )
    (seq_dir / "manifest.json").write_text(
        json.dumps({"tag": "t", "num_tokens": int(k.shape[1]),
                    "token_ids": token_ids})
    )


def test_load_sequence_roundtrip_and_ordering(tmp_path):
    torch.manual_seed(0)
    ls, n, h, d = 5, 7, 8, 128
    k = torch.randn(ls, n, h, d)
    v = torch.randn(ls, n, h, d)
    seq = tmp_path / hash_tokens(list(range(n)))
    write_sequence(seq, k, v, list(range(n)))

    assert num_layers(seq) == ls
    lk, lv = load_sequence(seq)
    assert lk.shape == (ls, n, h, d)
    # Axis 0 must be decoder order 0..L-1 despite reverse write order.
    assert torch.equal(lk, k)
    assert torch.equal(lv, v)


def write_tp_sequence(
    seq_dir: Path, k: torch.Tensor, v: torch.Tensor, world: int
) -> None:
    """Write ``[L, N, H, D]`` split across ``world`` per-rank head shards."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    ls, _, h, _ = k.shape
    hl = h // world
    for i in range(ls):
        for r in range(world):
            sl = slice(r * hl, (r + 1) * hl)
            kv = torch.stack((k[i, :, sl], v[i, :, sl]), dim=1)  # [N, 2, hl, D]
            save_file(
                {"kv": kv.contiguous()},
                str(seq_dir / f"model.layers.{i}.self_attn.attn.r{r:02d}.safetensors"),
            )


def test_tp_shards_reassemble_in_head_order(tmp_path):
    torch.manual_seed(1)
    ls, n, h, d = 3, 5, 8, 128
    k = torch.randn(ls, n, h, d)
    v = torch.randn(ls, n, h, d)
    seq = tmp_path / "tpseq"
    write_tp_sequence(seq, k, v, world=4)

    assert num_layers(seq) == ls  # counts layers, not shards
    lk, lv = load_sequence(seq)
    assert lk.shape == (ls, n, h, d)
    assert torch.equal(lk, k)  # heads concatenated back in rank order
    assert torch.equal(lv, v)


def test_sequence_dirs_and_manifest(tmp_path):
    k = torch.zeros(2, 4, 8, 128)
    for toks in ([1, 2, 3, 4], [5, 6, 7, 8]):
        write_sequence(tmp_path / hash_tokens(toks), k, k, toks)
    dirs = sequence_dirs(tmp_path)
    assert len(dirs) == 2
    assert load_manifest(dirs[0])["token_ids"] in ([1, 2, 3, 4], [5, 6, 7, 8])
    assert sequence_dirs(tmp_path / "missing") == []


def test_hash_is_stable_and_layout_agnostic():
    toks = [11, 22, 33, 44, 55]
    assert hash_tokens(toks) == hash_tokens(torch.tensor(toks))
    assert hash_tokens(toks) != hash_tokens(toks[:-1])


def test_layer_index_parsing():
    assert layer_index("model.layers.37.self_attn.attn") == 37
    assert layer_index("layers.0") == 0
    try:
        layer_index("model.embed_tokens")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on a name without a layer index")


def test_alignment_rules():
    bs = 16
    # Producer floor-aligns; consumer keeps a trailing token to forward.
    assert capture_align(70, bs) == 64
    assert serve_matched(70, bs) == 64
    assert serve_matched(64, bs) == 48  # exact multiple still yields a live token
    assert serve_matched(1, bs) == 0
