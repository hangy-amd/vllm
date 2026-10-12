"""GPU-to-GPU KVLink round-trip: bytes and seq_hash survive the NCCL hop.

Spawns a 2-rank producer/consumer pair (rank 0 on ``cuda:0``, rank 1 on
``cuda:1``) exactly as the live transfer connectors do, sends one known
``[L, N, H, D]`` K/V pair, and asserts the receiver reconstructs it byte-exactly
with the matching hash. Both ranks synthesize the reference from the same CPU RNG
seed, so the receiver can verify independently of the sender (the
``multi_gpu_comm.py`` pattern). Skipped without 2 visible GPUs.
"""

from __future__ import annotations

import json
import socket

import pytest
import torch
import torch.multiprocessing as mp

from cross_model_kv.capture_store import hash_tokens

_L, _N, _H, _D = 4, 32, 8, 128


def _reference_kv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1234)  # identical across processes
    k = torch.randn(_L, _N, _H, _D, generator=gen).to(device, dtype=torch.bfloat16)
    v = torch.randn(_L, _N, _H, _D, generator=gen).to(device, dtype=torch.bfloat16)
    return k, v


def _worker(rank: int, port: int, out_dir: str) -> None:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    from cross_model_kv.kv_link import CONSUMER_RANK, PRODUCER_RANK, KVLink

    link = KVLink.create("127.0.0.1", port, rank, device)
    seq_hash = hash_tokens(list(range(_N)))
    k, v = _reference_kv(device)

    if rank == PRODUCER_RANK:
        link.send_kv(seq_hash, k, v)
    else:
        assert rank == CONSUMER_RANK
        recv_hash, recv_k, recv_v = link.recv_kv()
        result = {
            "hash_match": recv_hash == seq_hash,
            "k_match": torch.equal(recv_k, k),
            "v_match": torch.equal(recv_v, v),
            "shape": list(recv_k.shape),
        }
        json.dump(result, open(f"{out_dir}/recv.json", "w"))


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs for a sender/receiver pair"
)
def test_kv_link_roundtrip(tmp_path):
    with socket.socket() as sock:
        sock.bind(("", 0))
        port = sock.getsockname()[1]

    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)

    res = json.load(open(tmp_path / "recv.json"))
    assert res["hash_match"], "seq_hash header did not survive the hop"
    assert res["k_match"], "received K differs from the sent K"
    assert res["v_match"], "received V differs from the sent V"
    assert res["shape"] == [_L, _N, _H, _D]
