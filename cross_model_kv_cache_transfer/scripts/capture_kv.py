"""Drive real Qwen3 prefill and capture stored K/V via KVCaptureConnector.

Runs prompt-only prefill (``max_tokens=1``) over calibration sequences with the
capture connector as a KV producer; the connector dumps per-layer ``[N, 2, H, D]``
stored (post-RoPE) caches to ``<capture_dir>/<tag>/``.

Example (ROCm, pin a connector-compatible attention backend)::

    VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \\
    .../python scripts/capture_kv.py --model Qwen/Qwen3-14B --tag 14b \\
        --capture-dir calib --num-seqs 64 --seq-len 1024

Run once per model of the directed pair (14b source, 32b target) with the same
``--capture-dir`` and matching ``--num-seqs``/``--seq-len`` so sequences align.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Make cross_model_kv importable here and in spawned vLLM workers.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_PKG_ROOT, os.environ.get("PYTHONPATH", "")) if p
)


def build_calibration_prompts(tokenizer, num_seqs: int, seq_len: int) -> list[list[int]]:
    """Tokenize FineWeb-Edu into fixed-length id chunks; fall back if offline."""
    texts: list[str] = []
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="sample-10BT",
            split="train", streaming=True,
        )
        for row in ds:
            texts.append(row["text"])
            if len(texts) >= num_seqs * 4:
                break
    except Exception as e:  # offline / dataset unavailable
        print(f"[capture] FineWeb-Edu unavailable ({e}); using wikitext fallback")
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t) > 200][: num_seqs * 4]

    ids: list[int] = []
    prompts: list[list[int]] = []
    for t in texts:
        ids.extend(tokenizer(t).input_ids)
        while len(ids) >= seq_len:
            prompts.append(ids[:seq_len])
            ids = ids[seq_len:]
            if len(prompts) >= num_seqs:
                return prompts
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="e.g. 14b or 32b")
    ap.add_argument("--capture-dir", default="calib")
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--prompts-file", default=None,
                    help="load calibration prompts (JSON list[list[int]]) instead "
                         "of building them; guarantees both models see identical seqs")
    ap.add_argument("--save-prompts", default=None,
                    help="write the built calibration prompts here for reuse")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--kv-cache-dtype", default="auto",
                    help="keep bf16/auto; FP8 KV would diverge the mapper")
    args = ap.parse_args()

    backend = os.environ.get("VLLM_ATTENTION_BACKEND", "<unset>")
    print(f"[capture] VLLM_ATTENTION_BACKEND={backend}")
    if backend == "ROCM_ATTN":
        raise SystemExit(
            "ROCM_ATTN is not validated for the connector transfer path; use "
            "ROCM_AITER_UNIFIED_ATTN or TRITON_ATTN."
        )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    import json as _json

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = _json.load(f)[: args.num_seqs]
        print(f"[capture] loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = build_calibration_prompts(tokenizer, args.num_seqs, args.seq_len)
        print(f"[capture] built {len(prompts)} calibration sequences "
              f"of {args.seq_len} tokens")
    if args.save_prompts:
        with open(args.save_prompts, "w") as f:
            _json.dump(prompts, f)
        print(f"[capture] saved prompts -> {args.save_prompts}")

    kv_cfg = KVTransferConfig(
        kv_connector="KVCaptureConnector",
        kv_connector_module_path="cross_model_kv.capture_connector",
        kv_role="kv_producer",
        kv_connector_extra_config={"capture_dir": args.capture_dir, "tag": args.tag},
    )
    llm = LLM(
        model=args.model,
        kv_transfer_config=kv_cfg,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        enforce_eager=True,
        # single-step prefill so the whole prompt's KV is captured at once
        enable_chunked_prefill=False,
        max_num_batched_tokens=max(args.seq_len, 2048),
    )

    sp = SamplingParams(max_tokens=1, temperature=0.0)
    llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp
    )

    out = Path(args.capture_dir) / args.tag
    files = sorted(glob.glob(str(out / "*" / "*.safetensors")))
    print(f"[capture] wrote {len(files)} layer shards under {out}")
    if files:
        from safetensors.torch import load_file

        kv = load_file(files[0])["kv"]
        print(f"[capture] sample shard {Path(files[0]).name}: kv shape {tuple(kv.shape)}")
        assert kv.ndim == 4 and kv.shape[1] == 2, f"unexpected KV layout {kv.shape}"
        print(f"[capture] n_kv_heads={kv.shape[2]} head_dim={kv.shape[3]}")


if __name__ == "__main__":
    main()
