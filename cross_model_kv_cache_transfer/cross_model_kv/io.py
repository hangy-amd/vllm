"""(De)serialisation of a fitted mapper: safetensors + JSON sidecar.

Layout on disk:
    <path>.safetensors  -- all weights/biases/selection tensors, plus the config
                           JSON embedded in the file's metadata block.
    <path>.json         -- the same config, human-readable alongside.

Tensor keys:
    selected.{t}        long [top_k]                 selected source layers
    key.weight.{t}      float32 [feature_dim, n_kv*d_h]
    key.bias.{t}        float32 [n_kv*d_h]
    val.weight.{t}      float32 [feature_dim, n_kv*d_h]
    val.bias.{t}        float32 [n_kv*d_h]
    selection_scores    float32 [num_target, num_source]   (optional diagnostic)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from cross_model_kv.config import MapperConfig
from cross_model_kv.mapper import LayerMap, RidgeKVMapper


def _stem(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix("") if path.suffix in (".safetensors", ".json") else path


def save_mapper(mapper: RidgeKVMapper, path: str | Path) -> Path:
    """Serialise ``mapper`` to ``<path>.safetensors`` (+ ``.json`` sidecar)."""
    stem = _stem(path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    cfg = mapper.cfg

    def _pack(x: torch.Tensor) -> torch.Tensor:
        return x.detach().float().cpu().contiguous()

    tensors: dict[str, torch.Tensor] = {}
    for t in range(cfg.num_target_layers):
        tensors[f"selected.{t}"] = mapper.selected[t].detach().to(torch.int64).cpu()
        tensors[f"key.weight.{t}"] = _pack(mapper.key_maps[t].weight)
        tensors[f"key.bias.{t}"] = _pack(mapper.key_maps[t].bias)
        tensors[f"val.weight.{t}"] = _pack(mapper.val_maps[t].weight)
        tensors[f"val.bias.{t}"] = _pack(mapper.val_maps[t].bias)
    if mapper.selection_scores is not None:
        tensors["selection_scores"] = _pack(mapper.selection_scores)

    config_json = json.dumps(cfg.to_dict())
    save_file(tensors, str(stem) + ".safetensors", metadata={"config": config_json})
    (stem.with_suffix(".json")).write_text(json.dumps(cfg.to_dict(), indent=2))
    return Path(str(stem) + ".safetensors")


def load_mapper(
    path: str | Path, device: torch.device | str = "cpu"
) -> RidgeKVMapper:
    """Load a mapper serialised by :func:`save_mapper`."""
    stem = _stem(path)
    st_path = str(stem) + ".safetensors"

    tensors = load_file(st_path, device=str(device))
    # Prefer the embedded config; fall back to the sidecar.
    import struct

    with open(st_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    meta = header.get("__metadata__", {})
    if "config" in meta:
        cfg = MapperConfig.from_dict(json.loads(meta["config"]))
    else:
        cfg = MapperConfig.from_dict(json.loads(stem.with_suffix(".json").read_text()))

    mapper = RidgeKVMapper(cfg)
    for t in range(cfg.num_target_layers):
        mapper.selected.append(tensors[f"selected.{t}"].to(torch.int64))
        mapper.key_maps.append(
            LayerMap(tensors[f"key.weight.{t}"], tensors[f"key.bias.{t}"])
        )
        mapper.val_maps.append(
            LayerMap(tensors[f"val.weight.{t}"], tensors[f"val.bias.{t}"])
        )
    mapper.selection_scores = tensors.get("selection_scores")
    return mapper
