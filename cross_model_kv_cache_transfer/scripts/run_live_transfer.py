"""Live GPU-to-GPU cross-model KV transfer driver (production path).

Runs the source (Qwen3-14B) and target (Qwen3-32B) models **concurrently** in two
processes, one GPU each, and streams each prefilled sequence's KV cache
GPU-to-GPU over a NCCL/RCCL :class:`~cross_model_kv.kv_link.KVLink` -- no disk,
no host round trip. The target skips its own prefill and decodes from the mapped
source cache.

Topology (same node):

    * producer -- source model on ``--producer-gpu`` (rank 0, launches the store)
    * consumer -- target model on ``--consumer-gpu`` (rank 1)

The orchestrator re-execs this script once per role with ``CUDA_VISIBLE_DEVICES``
pinned so each vLLM engine owns exactly one GPU (seen locally as ``cuda:0``). Both
roles read the **same** ordered prompt token-id list (a shared JSON of inputs, not
KV), so producer send #i pairs with consumer recv #i; the ``seq_hash`` header
asserts the pairing.

Example (ROCm)::

    VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \\
    python scripts/run_live_transfer.py \\
        --source-model Qwen/Qwen3-14B --target-model Qwen/Qwen3-32B \\
        --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \\
        --smoke-lengths 64,128 --with-baseline --out /tmp/live_out.json
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_PKG_ROOT, os.environ.get("PYTHONPATH", "")) if p
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def build_smoke_prompts(model: str, lengths: list[int]) -> list[list[int]]:
    """Tile a fixed passage into prompts of the requested token lengths."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    base = tok("The quick brown fox jumps over the lazy dog. " * 64).input_ids
    prompts = []
    for length in lengths:
        ids = (base * (length // len(base) + 1))[:length]
        prompts.append(ids)
    return prompts


# --------------------------------------------------------------------------- #
# Worker roles (each runs in its own CUDA_VISIBLE_DEVICES-pinned process)
# --------------------------------------------------------------------------- #
def _build_llm(model: str, kv_cfg, args):
    from vllm import LLM

    return LLM(
        model=model,
        kv_transfer_config=kv_cfg,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max(args.max_model_len, 2048),
    )


def run_producer(args) -> None:
    from vllm import SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    prompts = json.load(open(args.prompts_file))
    kv_cfg = KVTransferConfig(
        kv_connector="CrossModelKVSender",
        kv_connector_module_path="cross_model_kv.transfer_producer_connector",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "link_host": args.link_host, "link_port": args.link_port,
        },
    )
    llm = _build_llm(args.source_model, kv_cfg, args)
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    t0 = time.perf_counter()
    llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp)
    print(f"[producer] prefilled + sent {len(prompts)} sequences in "
          f"{time.perf_counter() - t0:.1f}s")


def run_consumer(args) -> None:
    from vllm import SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    prompts = json.load(open(args.prompts_file))
    kv_cfg = KVTransferConfig(
        kv_connector="CrossModelKVConnector",
        kv_connector_module_path="cross_model_kv.transfer_connector",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "mapper_path": args.mapper_path,
            "link_host": args.link_host, "link_port": args.link_port,
        },
    )
    llm = _build_llm(args.target_model, kv_cfg, args)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp)
    _dump_outputs(args.out, "transfer-live", args.target_model, outs)
    print(f"[consumer] wrote {len(outs)} outputs -> {args.out}")


def run_baseline(args) -> None:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    prompts = json.load(open(args.prompts_file))
    llm = _build_llm(args.target_model, None, args)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp)
    _dump_outputs(args.out, "baseline", args.target_model, outs)
    print(f"[baseline] wrote {len(outs)} outputs -> {args.out}")


def _dump_outputs(path: str, mode: str, model: str, outs) -> None:
    results = [
        {"idx": i, "gen_ids": list(o.outputs[0].token_ids),
         "gen": o.outputs[0].text[:200]}
        for i, o in enumerate(outs)
    ]
    json.dump({"mode": mode, "model": model, "results": results}, open(path, "w"))


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _spawn(role: str, gpu: int, out: str, args) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, os.path.abspath(__file__), "--role", role,
        "--source-model", args.source_model,
        "--target-model", args.target_model,
        "--mapper-path", args.mapper_path,
        "--prompts-file", args.prompts_file,
        "--link-host", args.link_host, "--link-port", str(args.link_port),
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--out", out,
    ]
    return subprocess.Popen(cmd, env=env)


def _divergence(live_path: str, base_path: str) -> None:
    live = {r["idx"]: r["gen_ids"] for r in json.load(open(live_path))["results"]}
    base = {r["idx"]: r["gen_ids"] for r in json.load(open(base_path))["results"]}
    diff = sum(1 for i in live if live[i] != base.get(i))
    total = len(live)
    print(f"[smoke] output divergence transfer vs native prefill: {diff}/{total}")
    if diff == 0:
        print("[smoke] WARNING: no divergence -- injection may not have driven "
              "the target (a silent prefill fallback yields identical greedy output)")
    else:
        print("[smoke] OK: mapped-KV injection provably drove the target model")


def orchestrate(args) -> None:
    if not args.prompts_file:
        lengths = [int(x) for x in args.smoke_lengths.split(",") if x]
        prompts = build_smoke_prompts(args.source_model, lengths)
        args.prompts_file = os.path.join(
            args.work_dir or "/tmp", "live_transfer_prompts.json"
        )
        json.dump(prompts, open(args.prompts_file, "w"))
        print(f"[orch] built {len(prompts)} smoke prompts "
              f"(lengths {lengths}) -> {args.prompts_file}")
    if args.link_port == 0:
        args.link_port = _free_port()
    print(f"[orch] KVLink {args.link_host}:{args.link_port}  "
          f"producer=gpu{args.producer_gpu} consumer=gpu{args.consumer_gpu}")

    prod_out = os.path.join(args.work_dir or "/tmp", "live_producer.json")
    procs = {
        "producer": _spawn("producer", args.producer_gpu, prod_out, args),
        "consumer": _spawn("consumer", args.consumer_gpu, args.out, args),
    }
    try:
        rc = {name: p.wait(timeout=args.timeout) for name, p in procs.items()}
    except subprocess.TimeoutExpired:
        for p in procs.values():
            p.kill()
        raise SystemExit("[orch] live transfer timed out; killed both processes")
    if any(code != 0 for code in rc.values()):
        raise SystemExit(f"[orch] worker exit codes {rc}")
    print(f"[orch] live transfer complete -> {args.out}")

    if args.with_baseline:
        base_out = os.path.join(args.work_dir or "/tmp", "live_baseline.json")
        proc = _spawn("baseline", args.consumer_gpu, base_out, args)
        if proc.wait(timeout=args.timeout) != 0:
            raise SystemExit("[orch] baseline run failed")
        _divergence(args.out, base_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["producer", "consumer", "baseline"],
                    default=None, help="internal: set by the orchestrator")
    ap.add_argument("--source-model", default="Qwen/Qwen3-14B")
    ap.add_argument("--target-model", default="Qwen/Qwen3-32B")
    ap.add_argument("--mapper-path", required=True)
    ap.add_argument("--prompts-file", default="")
    ap.add_argument("--smoke-lengths", default="64,128",
                    help="token lengths to synthesize when no --prompts-file given")
    ap.add_argument("--out", default="/tmp/live_out.json")
    ap.add_argument("--work-dir", default="/tmp")
    ap.add_argument("--link-host", default="127.0.0.1")
    ap.add_argument("--link-port", type=int, default=0)
    ap.add_argument("--producer-gpu", type=int, default=0)
    ap.add_argument("--consumer-gpu", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--with-baseline", action="store_true",
                    help="also run native-prefill baseline and report divergence")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    if args.role == "producer":
        run_producer(args)
    elif args.role == "consumer":
        run_consumer(args)
    elif args.role == "baseline":
        run_baseline(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
