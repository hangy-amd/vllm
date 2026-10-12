"""Live (GPU-to-GPU) KV producer connector for cross-model transfer.

The offline counterpart, :class:`~cross_model_kv.capture_connector.KVCaptureConnector`,
dumps the source model's post-RoPE K/V to disk for offline mapper fitting. This
connector is the *serving* producer: it runs the source model (Qwen3-14B) as a
``kv_producer`` and streams each prefilled sequence's whole-cache K/V straight to
the target-model process over a NCCL/RCCL :class:`~cross_model_kv.kv_link.KVLink`
-- no disk, no host round trip.

It reuses the capture connector's scheduler-side logic and per-layer slot
extraction verbatim; only the sink changes. Per-layer extracts are buffered on
device across ``save_kv_layer`` calls and, once the whole forward's layers are in,
:meth:`wait_for_save` stacks them to ``[L, N, H, D]`` and sends one sequence at a
time to the consumer (rank 1).

Register out-of-tree::

    --kv-transfer-config '{"kv_connector":"CrossModelKVSender",
      "kv_connector_module_path":"cross_model_kv.transfer_producer_connector",
      "kv_role":"kv_producer",
      "kv_connector_extra_config":{"link_host":"127.0.0.1","link_port":29555}}'

Assumes tensor-parallel size 1 for the source model (the KVLink is a 2-rank
producer/consumer group; TP>1 would need per-rank links, out of scope here).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from cross_model_kv.capture_connector import (
    CaptureConnectorMetadata,
    KVCaptureConnector,
)
from cross_model_kv.capture_store import layer_index
from cross_model_kv.kv_link import PRODUCER_RANK, KVLink
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(f"vllm.{__name__}")

Tensor = torch.Tensor


class CrossModelKVSender(KVCaptureConnector):
    """Producer connector that NCCL-sends whole-sequence KV to the consumer.

    Inherits the scheduler-side behaviour of :class:`KVCaptureConnector`
    (never skip prefill; build one :class:`CaptureReqMeta` per new request) and
    its :meth:`_extract` slot gather; overrides only the worker-side sink.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        # Bypass KVCaptureConnector.__init__ (which creates on-disk capture
        # directories) and initialise the connector base directly.
        KVConnectorBase_V1.__init__(
            self, vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config
        )
        self._block_size = vllm_config.cache_config.block_size
        extra = self._kv_transfer_config.kv_connector_extra_config
        self._link_host = extra.get("link_host", "127.0.0.1")
        self._link_port = int(extra["link_port"])

        self._link: KVLink | None = None
        # seq_hash -> {layer_index -> [N, 2, H, D]} accumulated across layers.
        self._buf: dict[str, dict[int, Tensor]] = {}
        logger.info(
            "CrossModelKVSender[%s] link=%s:%d",
            role, self._link_host, self._link_port,
        )

    def _ensure_link(self, device: torch.device) -> None:
        if self._link is None:
            self._link = KVLink.create(
                self._link_host, self._link_port, PRODUCER_RANK, device
            )

    # ---- worker side: buffer per layer, send per sequence -------------------
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        meta = self._get_connector_metadata()
        assert isinstance(meta, CaptureConnectorMetadata)
        if not meta.requests:
            return
        self._ensure_link(kv_layer.device)
        li = layer_index(layer_name)
        for req in meta.requests:
            kv = self._extract(kv_layer, req.slot_mapping, attn_metadata)
            self._buf.setdefault(req.seq_hash, {})[li] = kv

    def wait_for_save(self) -> None:
        if not self._buf:
            return
        assert self._link is not None
        # Insertion order of _buf mirrors scheduled-request order (populated
        # during layer 0's save_kv_layer), which the consumer receives in the
        # same order; the seq_hash header lets the consumer assert the pairing.
        for seq_hash, layers in self._buf.items():
            ordered = [layers[i] for i in sorted(layers)]
            kv = torch.stack(ordered)          # [L, N, 2, H, D]
            k, v = kv[:, :, 0], kv[:, :, 1]    # [L, N, H, D] each
            self._link.send_kv(seq_hash, k, v)
            logger.info(
                "sent source KV over NCCL: %d layers, %d tokens (req hash %s)",
                k.shape[0], k.shape[1], seq_hash[:8],
            )
        self._buf.clear()
