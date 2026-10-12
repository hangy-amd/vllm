"""Prefill-skip latency benchmark for cross-model KV transfer (M5).

Times single-request time-to-first-token on the target model (Qwen3-32B) across
prompt lengths, in two modes:

* ``baseline``  — 32B prefills the whole prompt itself.
* ``transfer``  — 32B injects the mapped 14B cache and skips prefill of the
  block-aligned prefix (mapping + scatter replaces the O(N^2) prefill).

Prefix caching is disabled so every iteration genuinely re-prefills / re-injects;
a warmup iteration primes the OS page cache for the source shards so timed
transfer iterations measure compute + H2D movement, not cold disk reads.

    speedup(L) = baseline_ttft(L) / transfer_ttft(L)

Example (ROCm)::

    VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \\
    CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \\
    python scripts/bench_latency.py --model Qwen/Qwen3-32B --mode transfer \\
        --prompts-file lat/prompts.json --lens-file lat/lens.json \\
        --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \\
        --source-cache-dir lat/14b --out lat/ttft_transfer.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_PKG_ROOT, os.environ.get("PYTHONPATH", "")) if p
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["baseline", "transfer"], required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--lens-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapper-path", default="")
    ap.add_argument("--source-cache-dir", default="")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=20000)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    prompts: list[list[int]] = json.load(open(args.prompts_file))
    lens: list[int] = json.load(open(args.lens_file))

    kv_cfg = None
    if args.mode == "transfer":
        kv_cfg = KVTransferConfig(
            kv_connector="CrossModelKVConnector",
            kv_connector_module_path="cross_model_kv.transfer_connector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "mapper_path": args.mapper_path,
                "source_cache_dir": args.source_cache_dir,
            },
        )

    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        max_num_batched_tokens=args.max_model_len,
    )

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    def ttft(ids: list[int]) -> float:
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
        return time.perf_counter() - t0

    rows = []
    for L, ids in zip(lens, prompts):
        for _ in range(args.warmup):
            ttft(ids)
        samples = sorted(ttft(ids) for _ in range(args.iters))
        median = samples[len(samples) // 2]
        rows.append({"len": L, "median_s": median, "samples_s": samples})
        print(f"[bench:{args.mode}] len={L:>6} median_ttft={median*1e3:8.1f} ms")

    json.dump({"mode": args.mode, "model": args.model, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"[bench:{args.mode}] wrote {args.out}")


if __name__ == "__main__":
    main()
