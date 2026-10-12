"""Live GPU-to-GPU KV transport for cross-model transfer.

The offline path (:class:`~cross_model_kv.capture_connector.KVCaptureConnector`)
writes the source cache to disk and the consumer reads it back host-side. This
module is the *production* alternative: the source (14B) and target (32B) models
run concurrently in separate processes, and a sequence's KV crosses the
interconnect **GPU-to-GPU** (RCCL/NCCL peer DMA) with no disk and no host copy.

It reuses vLLM's own side-channel primitives rather than a second global
``torch.distributed`` group (which would collide with each model's tensor-parallel
group):

    * :class:`~vllm.distributed.utils.StatelessProcessGroup` -- a TCP-store
      rendezvous used only for the small ``{seq_hash, shape, dtype}`` header;
    * :class:`~vllm.distributed.device_communicators.pynccl.PyNcclCommunicator`
      -- the NCCL/RCCL data plane for the KV tensors themselves.

Rank 0 is the producer (source model), rank 1 the consumer (target model). The
producer launches the store server, so it must be reachable at ``host:port``.
Sends and receives are ordered: producer send #i pairs with consumer recv #i, and
the ``seq_hash`` header lets the consumer assert the pairing is correct.
"""

from __future__ import annotations

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

logger = init_logger(f"vllm.{__name__}")

Tensor = torch.Tensor

PRODUCER_RANK = 0
CONSUMER_RANK = 1

_STR_TO_DTYPE = {
    str(dt): dt
    for dt in (torch.float32, torch.float16, torch.bfloat16)
}


class KVLink:
    """A 2-rank NCCL point-to-point channel for whole-sequence KV caches.

    Build one per worker with :meth:`create`; the constructor blocks until both
    ranks have rendezvoused (the ``PyNcclCommunicator`` exchanges its unique id
    over the store). ``k``/``v`` are the stacked ``[L, N, H, D]`` caches.
    """

    def __init__(
        self,
        spg: StatelessProcessGroup,
        comm: PyNcclCommunicator,
        rank: int,
        device: torch.device,
    ):
        self._spg = spg
        self._comm = comm
        self.rank = rank
        self.device = device

    @classmethod
    def create(
        cls,
        host: str,
        port: int,
        rank: int,
        device: torch.device | str,
        world_size: int = 2,
        store_timeout: int = 300,
    ) -> "KVLink":
        device = torch.device(device)
        spg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size,
            store_timeout=store_timeout,
        )
        comm = PyNcclCommunicator(spg, device=device)
        if not comm.available:
            raise RuntimeError(
                "PyNcclCommunicator is unavailable (NCCL/RCCL library not "
                "loaded); cannot establish the GPU-to-GPU KV link"
            )
        logger.info(
            "KVLink up: rank=%d/%d device=%s peer=%s:%d",
            rank, world_size, device, host, port,
        )
        return cls(spg, comm, rank, device)

    # ---- producer (rank 0) --------------------------------------------------
    def send_kv(self, seq_hash: str, k: Tensor, v: Tensor) -> None:
        """Send one sequence's ``[L, N, H, D]`` K and V to the consumer."""
        assert self.rank == PRODUCER_RANK, "send_kv is a producer-only op"
        k = k.contiguous()
        v = v.contiguous()
        header = {
            "seq_hash": seq_hash,
            "k_shape": tuple(k.shape),
            "v_shape": tuple(v.shape),
            "dtype": str(k.dtype),
        }
        self._spg.send_obj(header, dst=CONSUMER_RANK)
        self._comm.send(k, dst=CONSUMER_RANK)
        self._comm.send(v, dst=CONSUMER_RANK)
        torch.cuda.synchronize(self.device)

    # ---- consumer (rank 1) --------------------------------------------------
    def recv_kv(self) -> tuple[str, Tensor, Tensor]:
        """Receive one sequence's K and V; returns ``(seq_hash, k, v)``."""
        assert self.rank == CONSUMER_RANK, "recv_kv is a consumer-only op"
        header = self._spg.recv_obj(src=PRODUCER_RANK)
        dtype = _STR_TO_DTYPE[header["dtype"]]
        k = torch.empty(header["k_shape"], dtype=dtype, device=self.device)
        v = torch.empty(header["v_shape"], dtype=dtype, device=self.device)
        self._comm.recv(k, src=PRODUCER_RANK)
        self._comm.recv(v, src=PRODUCER_RANK)
        torch.cuda.synchronize(self.device)
        return header["seq_hash"], k, v
