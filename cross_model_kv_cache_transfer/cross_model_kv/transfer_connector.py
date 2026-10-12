"""Serving-time cross-model KV transfer connector (Phase B / M3).

A *consumer* :class:`KVConnectorBase_V1` for the target model (e.g. Qwen3-32B).
For a prompt whose source-model (e.g. Qwen3-14B) cache is available on shared
storage, it:

1. tells the scheduler the whole prompt is already computed
   (:meth:`get_num_new_matched_tokens`) so the target **skips its own prefill**;
2. on the worker, loads the source cache, strips source RoPE into content space,
   applies the fitted closed-form mapper per target layer, re-applies target
   RoPE, and scatters the mapped K/V into the target's paged buffer by slot.

The source cache is produced by :class:`~cross_model_kv.capture_connector.KVCaptureConnector`
(run the source model as a ``kv_producer`` into ``source_cache_dir``); this is
the shared-filesystem MVP transport the DESIGN recommends. Both sides key on the
full prompt hash, so a hit means "this exact prompt was prefilled by the source".

Register out-of-tree (no vLLM core edit)::

    --kv-transfer-config '{"kv_connector":"CrossModelKVConnector",
      "kv_connector_module_path":"cross_model_kv.transfer_connector",
      "kv_role":"kv_consumer",
      "kv_connector_extra_config":{
        "mapper_path":"cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors",
        "source_cache_dir":"calib/14b"}}'
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from cross_model_kv.capture_store import (
    hash_tokens,
    layer_index,
    load_sequence,
    serve_matched,
)
from cross_model_kv.io import load_mapper
from cross_model_kv.mapper import RidgeKVMapper
from cross_model_kv.rope import RopeCodec
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

Tensor = torch.Tensor


def _tp_rank_world() -> tuple[int, int]:
    """(rank, world_size) of the tensor-parallel group, or (0, 1) if unset."""
    try:
        return get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size()
    except Exception:
        return 0, 1


@dataclass
class TransferReqMeta:
    seq_hash: str
    num_tokens: int          # block-aligned prompt length being injected
    slot_mapping: torch.Tensor

    @staticmethod
    def make(seq_hash: str, block_ids: list[int], block_size: int,
             num_tokens: int) -> "TransferReqMeta":
        blocks = torch.tensor(block_ids, dtype=torch.long)
        offsets = torch.arange(block_size)
        slot = (offsets.reshape(1, -1) + blocks.reshape(-1, 1) * block_size)
        slot = slot.flatten()[:num_tokens]
        return TransferReqMeta(seq_hash, num_tokens, slot)


@dataclass
class TransferConnectorMetadata(KVConnectorMetadata):
    requests: list[TransferReqMeta] = field(default_factory=list)


class CrossModelKVConnector(KVConnectorBase_V1):
    """Consumer connector: inject a mapped source cache instead of prefilling."""

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
        self._mapper_path = extra.get("mapper_path", "")
        self._source_cache_dir = extra.get("source_cache_dir", "")

        # Live GPU-to-GPU transport: when a link port is set, receive the source
        # cache over NCCL from the producer instead of reading it from disk.
        self._link_host = extra.get("link_host", "127.0.0.1")
        self._link_port = int(extra.get("link_port", 0))
        self._live = self._link_port > 0
        self._link: Any = None

        # Scheduler-side: requests whose blocks were allocated to receive KV.
        self._need_load: dict[str, Request] = {}

        # Worker-side: lazily materialised on first load.
        self._mapper: RidgeKVMapper | None = None
        self._src_codec: RopeCodec | None = None
        self._tgt_codec: RopeCodec | None = None
        self._layer_to_target: dict[str, int] | None = None
        logger.info(
            "CrossModelKVConnector[%s] mapper=%s source=%s",
            role, self._mapper_path,
            f"link {self._link_host}:{self._link_port}" if self._live
            else self._source_cache_dir,
        )

    # ---- shared helpers -----------------------------------------------------
    def _source_seq_dir(self, token_ids: list[int]) -> str:
        return os.path.join(self._source_cache_dir, hash_tokens(token_ids))

    def _has_source(self, token_ids: list[int]) -> bool:
        # In live mode the driver guarantees a matching producer send for every
        # submitted prompt, so there is no disk directory to stat.
        if self._live:
            return True
        return bool(self._source_cache_dir) and os.path.isdir(
            self._source_seq_dir(token_ids)
        )

    # ---- scheduler side -----------------------------------------------------
    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        token_ids = list(request.prompt_token_ids or [])
        if not self._has_source(token_ids):
            return 0, False
        num_matched = serve_matched(len(token_ids), self._block_size)
        new_tokens = num_matched - num_computed_tokens
        if new_tokens <= 0:
            # Nothing to serve (e.g. the whole prompt is a single block, which
            # the target must run itself to produce logits). Falls back to a
            # normal prefill; note block_size can be large (64 on ROCm AITER).
            return 0, False
        logger.info(
            "cross-model KV hit: skipping prefill of %d tokens for req %s",
            new_tokens, request.request_id,
        )
        # Synchronous injection in the next forward's start_load_kv.
        return new_tokens, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        if num_external_tokens > 0:
            self._need_load[request.request_id] = request

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = TransferConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            if new_req.req_id not in self._need_load:
                continue
            token_ids = list(new_req.prompt_token_ids or [])
            num_matched = serve_matched(len(token_ids), self._block_size)
            meta.requests.append(
                TransferReqMeta.make(
                    hash_tokens(token_ids), new_req.block_ids[0],
                    self._block_size, num_matched,
                )
            )
        self._need_load.clear()
        return meta

    # ---- worker side --------------------------------------------------------
    def _ensure_loaded(self, device: torch.device) -> None:
        if self._mapper is not None:
            return
        if not self._mapper_path:
            raise ValueError("CrossModelKVConnector requires 'mapper_path'")
        mapper = load_mapper(self._mapper_path, device=device)
        cfg = mapper.cfg
        self._mapper = mapper
        self._src_codec = RopeCodec(
            cfg.head_dim, cfg.rotary_dim, cfg.source_rope_theta, device=device
        )
        self._tgt_codec = RopeCodec(
            cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta, device=device
        )
        logger.info(
            "loaded mapper %s -> %s (%dL->%dL, top_k=%d)",
            cfg.source_model, cfg.target_model, cfg.num_source_layers,
            cfg.num_target_layers, cfg.top_k,
        )

    def _ensure_link(self, device: torch.device) -> None:
        if self._link is None:
            from cross_model_kv.kv_link import CONSUMER_RANK, KVLink

            self._link = KVLink.create(
                self._link_host, self._link_port, CONSUMER_RANK, device
            )

    def _map_sequence(
        self, src_k: Tensor, src_v: Tensor, positions: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Map a source cache to per-target-layer *stored* K/V.

        ``src_k`` / ``src_v`` are ``[L_s, N, H, D]`` (K post-RoPE). Returns two
        lists indexed by target layer, each entry ``[N, H, D]``: K carries the
        target's RoPE, V is positionless. This is pure math — no vLLM runtime —
        so it is unit-testable in isolation.
        """
        assert self._mapper is not None
        src_k_c = self._src_codec.strip_at(src_k, positions)
        out_k, out_v = [], []
        for t in range(self._mapper.cfg.num_target_layers):
            k_c, v = self._mapper.transfer_layer(src_k_c, src_v, t)
            out_k.append(self._tgt_codec.apply_at(k_c, positions))
            out_v.append(v)
        return out_k, out_v

    def _target_layer_map(
        self, forward_context: "ForwardContext"
    ) -> dict[str, int]:
        """Map each attention layer name to its 0-based target layer index."""
        if self._layer_to_target is None:
            names = [
                name for name, layer in forward_context.no_compile_layers.items()
                if getattr(layer, "kv_cache", None) is not None
            ]
            ordered = sorted(names, key=layer_index)
            self._layer_to_target = {name: i for i, name in enumerate(ordered)}
        return self._layer_to_target

    def _inject(
        self,
        layer_cache: Tensor,
        k: Tensor,
        v: Tensor,
        slot_mapping: Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """Scatter mapped ``k``/``v`` (``[N, H, D]``) into the paged buffer.

        Mirrors :meth:`KVCaptureConnector._extract`: the HND buffer packs K and V
        into the last dim, so we re-concatenate before the slot scatter.
        """
        slot_mapping = slot_mapping.to(layer_cache.device)
        if isinstance(attn_metadata, MLACommonMetadata):
            flat = layer_cache.reshape(-1, layer_cache.shape[-1])
            flat[slot_mapping] = torch.cat((k, v), dim=-1)
            return
        block_idxs = slot_mapping // self._block_size
        offsets = slot_mapping % self._block_size
        layer_cache[block_idxs, :, offsets] = torch.cat((k, v), dim=-1)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, TransferConnectorMetadata)
        if not meta.requests:
            return
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("start_load_kv called with attn_metadata=None")
            return

        layer_map = self._target_layer_map(forward_context)
        layers = forward_context.no_compile_layers
        rank, world = _tp_rank_world()

        for req in meta.requests:
            # Peek device from any attention layer's kv cache.
            probe = next(
                getattr(layers[nm], "kv_cache")
                for nm in layer_map
            )
            device = probe.device
            self._ensure_loaded(device)

            if self._live:
                self._ensure_link(device)
                seq_hash, src_k, src_v = self._link.recv_kv()
                if seq_hash != req.seq_hash:
                    raise RuntimeError(
                        f"KVLink ordering mismatch: received {seq_hash[:8]} "
                        f"but expected {req.seq_hash[:8]}"
                    )
            else:
                src_dir = os.path.join(self._source_cache_dir, req.seq_hash)
                src_k, src_v = load_sequence(src_dir, device)
            n = req.num_tokens
            src_k, src_v = src_k[:, :n], src_v[:, :n]
            positions = torch.arange(n, device=device)
            mapped_k, mapped_v = self._map_sequence(src_k, src_v, positions)

            for layer_name, t in layer_map.items():
                layer_cache = layers[layer_name].kv_cache
                if isinstance(layer_cache, (list, tuple)):
                    layer_cache = layer_cache[0]
                md = (
                    attn_metadata[layer_name]
                    if isinstance(attn_metadata, dict) else attn_metadata
                )
                k, v = mapped_k[t], mapped_v[t]
                if world > 1 and not isinstance(md, MLACommonMetadata):
                    # This rank owns only its slice of the KV heads.
                    hl = layer_cache.shape[1]
                    k = k[:, rank * hl:(rank + 1) * hl, :]
                    v = v[:, rank * hl:(rank + 1) * hl, :]
                self._inject(
                    layer_cache,
                    k.to(layer_cache.dtype),
                    v.to(layer_cache.dtype),
                    req.slot_mapping,
                    md,
                )
            logger.info(
                "injected mapped KV for %d tokens (req hash %s)",
                n, req.seq_hash[:8],
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata, **kwargs: Any,
    ) -> None:
        return

    def wait_for_save(self) -> None:
        return
