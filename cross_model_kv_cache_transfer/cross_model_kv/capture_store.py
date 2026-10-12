"""Shared IO for the on-disk capture format (``KVCaptureConnector`` output).

Both the offline fit (``scripts/fit_mapper.py``) and the serving-time consumer
(:class:`~cross_model_kv.transfer_connector.CrossModelKVConnector`) read the
per-sequence caches written by :class:`~cross_model_kv.capture_connector.KVCaptureConnector`,
so the format, the token hashing, and the block-alignment rules live here in one
vLLM-free place to keep producer and consumer in lock-step.

On disk::

    <tag_dir>/<seq_hash>/<layer_name>.safetensors        # single-GPU capture
    <tag_dir>/<seq_hash>/<layer_name>.r<rank>.safetensors # tensor-parallel capture
    <tag_dir>/<seq_hash>/manifest.json                    # {tag, num_tokens, token_ids}

Each shard holds ``{"kv": [N, 2, H_local, D]}``. Under tensor parallelism a layer
is split across ``world_size`` shards, each carrying ``num_kv_heads // world_size``
of the heads; :func:`load_sequence` concatenates them in rank order to recover the
full ``[N, 2, H, D]`` cache. Row ``i`` is token position ``i`` (each prompt is
prefilled from position 0). ``kv[:, 0]`` is K (post-RoPE), ``kv[:, 1]`` is V.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

Tensor = torch.Tensor

_LAYER_RE = re.compile(r"layers\.(\d+)")
_RANK_RE = re.compile(r"\.r(\d+)\.safetensors$")


def hash_tokens(token_ids: list[int] | Tensor) -> str:
    """Stable content hash of a prompt's *full* token id list.

    Producer and consumer both key on the whole prompt (not a block-aligned
    prefix), so the lookup is independent of either side's block size. Uses
    blake2b over the int64 byte representation to stay free of any vLLM import.
    """
    if isinstance(token_ids, torch.Tensor):
        arr = token_ids.detach().cpu().to(torch.int64)
    else:
        arr = torch.tensor(list(token_ids), dtype=torch.int64)
    return hashlib.blake2b(arr.numpy().tobytes(), digest_size=16).hexdigest()


def capture_align(num_tokens: int, block_size: int) -> int:
    """Whole-block token count the capture connector stores: ``floor``-aligned."""
    return (num_tokens // block_size) * block_size


def serve_matched(num_tokens: int, block_size: int) -> int:
    """Block-aligned prompt length the consumer skips prefill for.

    Aligns ``num_tokens - 1`` down to a block so at least one trailing token is
    still forwarded through the target model (mirrors vLLM's example connectors);
    a request with nothing left to compute cannot proceed to decode.
    """
    return ((num_tokens - 1) // block_size) * block_size


def layer_index(layer_name: str) -> int:
    """Integer decoder-layer index parsed from a vLLM attention layer name.

    e.g. ``model.layers.37.self_attn.attn`` -> 37. Raises if absent so callers
    never silently misorder layers.
    """
    m = _LAYER_RE.search(layer_name)
    if m is None:
        raise ValueError(f"cannot parse a layer index from {layer_name!r}")
    return int(m.group(1))


def sequence_dirs(tag_dir: str | Path) -> list[Path]:
    """Every captured-sequence directory under ``tag_dir``, sorted by hash."""
    tag_dir = Path(tag_dir)
    return sorted(p for p in tag_dir.iterdir() if p.is_dir()) if tag_dir.is_dir() else []


def _shard_rank(path: Path) -> int:
    m = _RANK_RE.search(path.name)
    return int(m.group(1)) if m else 0


def _layers_by_index(seq_dir: Path) -> dict[int, list[Path]]:
    """Group shard files by decoder-layer index, each value sorted by TP rank."""
    layers: dict[int, list[Path]] = {}
    for p in seq_dir.glob("*.safetensors"):
        layers.setdefault(layer_index(p.name), []).append(p)
    for paths in layers.values():
        paths.sort(key=_shard_rank)
    return layers


def load_sequence(
    seq_dir: str | Path, device: torch.device | str = "cpu"
) -> tuple[Tensor, Tensor]:
    """Load one captured sequence as stacked ``[L, N, H, D]`` K and V caches.

    Layers are ordered by their parsed decoder index (axis 0 is ``0..L-1``
    regardless of filesystem order); within a layer, tensor-parallel rank shards
    are concatenated in rank order along the KV-head axis.
    """
    seq_dir = Path(seq_dir)
    layers = _layers_by_index(seq_dir)
    if not layers:
        raise FileNotFoundError(f"no layer shards under {seq_dir}")
    keys, vals = [], []
    for idx in sorted(layers):
        shards = [load_file(str(p), device=str(device))["kv"] for p in layers[idx]]
        kv = shards[0] if len(shards) == 1 else torch.cat(shards, dim=2)  # heads
        keys.append(kv[:, 0])
        vals.append(kv[:, 1])
    return torch.stack(keys), torch.stack(vals)


def load_manifest(seq_dir: str | Path) -> dict:
    """Read a sequence's ``manifest.json`` (token_ids, num_tokens, tag)."""
    return json.loads((Path(seq_dir) / "manifest.json").read_text())


def num_layers(seq_dir: str | Path) -> int:
    return len(_layers_by_index(Path(seq_dir)))
