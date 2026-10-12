"""Offline KV-capture connector (Phase A of cross-model KV transfer).

A store-only :class:`KVConnectorBase_V1` that dumps the *stored* (post-RoPE)
per-layer K/V for every prefilled prompt to disk, so the mapper can be fit
offline on real Qwen3 caches. It never loads or skips prefill.

Register out-of-tree (no vllm core edit) via ``kv_connector_module_path``::

    --kv-transfer-config '{"kv_connector":"KVCaptureConnector",
      "kv_connector_module_path":"cross_model_kv.capture_connector",
      "kv_role":"kv_producer",
      "kv_connector_extra_config":{"capture_dir":"calib","tag":"14b"}}'

On disk::

    <capture_dir>/<tag>/<seq_hash>/<layer_name>.safetensors   # kv: [N, 2, H, D]
    <capture_dir>/<tag>/<seq_hash>/manifest.json              # token_ids, N

Row ``i`` of every layer tensor is token position ``i`` (each prompt is
prefilled from position 0 with no prefix), so RoPE can be stripped downstream
without storing positions explicitly. ``kv[:, 0]`` is K, ``kv[:, 1]`` is V.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch

from cross_model_kv.capture_store import hash_tokens
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(f"vllm.{__name__}")


def _tp_rank_world() -> tuple[int, int]:
    """(rank, world_size) of the tensor-parallel group, or (0, 1) if unset.

    Under TP each worker holds only ``num_kv_heads // world_size`` KV heads, so
    the captured shards are per-rank slices reassembled downstream by
    :func:`cross_model_kv.capture_store.load_sequence`.
    """
    try:
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        return 0, 1


@dataclass
class CaptureReqMeta:
    seq_hash: str
    token_ids: torch.Tensor
    slot_mapping: torch.Tensor

    @staticmethod
    def make(token_ids: list[int], block_ids: list[int], block_size: int
             ) -> "CaptureReqMeta":
        # Key on the whole prompt so the consumer (which may align differently)
        # finds this sequence; store only whole blocks, so row i is position i.
        seq_hash = hash_tokens(token_ids)
        n = (len(token_ids) // block_size) * block_size
        tok = torch.tensor(token_ids[:n], dtype=torch.long)
        blocks = torch.tensor(block_ids, dtype=torch.long)
        offsets = torch.arange(block_size)
        slot = (offsets.reshape(1, -1) + blocks.reshape(-1, 1) * block_size)
        slot = slot.flatten()[:n]
        return CaptureReqMeta(seq_hash, tok, slot)


@dataclass
class CaptureConnectorMetadata(KVConnectorMetadata):
    requests: list[CaptureReqMeta] = field(default_factory=list)


class KVCaptureConnector(KVConnectorBase_V1):
    """Store-only connector: dumps per-layer stored K/V for offline fitting."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._capture_dir = extra.get("capture_dir", "calib")
        self._tag = extra.get("tag", "model")
        self._out = os.path.join(self._capture_dir, self._tag)
        # Requests scheduled to prefill this step (scheduler side).
        self._pending: set[str] = set()
        if role == KVConnectorRole.WORKER:
            os.makedirs(self._out, exist_ok=True)
        logger.info("KVCaptureConnector[%s] -> %s", self._tag, self._out)

    # ---- scheduler side -----------------------------------------------------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        # Never serve from external cache: we want a real prefill to populate KV.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        return

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = CaptureConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or []
            if len(token_ids) < self._block_size:
                continue
            meta.requests.append(
                CaptureReqMeta.make(
                    list(token_ids), new_req.block_ids[0], self._block_size
                )
            )
        return meta

    # ---- worker side --------------------------------------------------------
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, CaptureConnectorMetadata)
        rank, world = _tp_rank_world()
        for req in meta.requests:
            kv = self._extract(kv_layer, req.slot_mapping, attn_metadata)
            seq_dir = os.path.join(self._out, req.seq_hash)
            os.makedirs(seq_dir, exist_ok=True)
            # One shard per TP rank; the reader concatenates them on the KV-head
            # axis to recover the full [N, 2, H, D] cache.
            shard = f"{layer_name}.r{rank:02d}.safetensors" if world > 1 \
                else f"{layer_name}.safetensors"
            safetensors.torch.save_file(
                {"kv": kv.detach().cpu().contiguous()},
                os.path.join(seq_dir, shard),
            )
            manifest = os.path.join(seq_dir, "manifest.json")
            if rank == 0 and not os.path.exists(manifest):
                with open(manifest, "w") as f:
                    json.dump(
                        {"tag": self._tag, "num_tokens": int(kv.shape[0]),
                         "token_ids": req.token_ids.tolist()}, f)

    def wait_for_save(self) -> None:
        return

    def _extract(
        self,
        layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Gather per-token K/V by slot -> ``[N, 2, H, D]`` (non-MLA).

        The per-layer buffer is the HND view ``[num_blocks, H, block_size,
        2*head_dim]`` used by the connector-compatible ROCm/flash backends, with
        K and V packed into the last dim (paper DESIGN sec 4.2). We split them
        apart and stack on a leading K/V axis so the stored shape is layout-
        agnostic for the downstream mapper.
        """
        slot_mapping = slot_mapping.to(layer.device)
        if isinstance(attn_metadata, MLACommonMetadata):
            return layer.reshape(-1, layer.shape[-1])[slot_mapping, ...]
        block_idxs = slot_mapping // self._block_size
        offsets = slot_mapping % self._block_size
        raw = layer[block_idxs, :, offsets]          # [N, H, 2*D]
        d = raw.shape[-1] // 2
        return torch.stack((raw[..., :d], raw[..., d:]), dim=1)  # [N, 2, H, D]
