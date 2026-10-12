"""Generative accuracy eval for cross-model KV transfer (M5).

Runs the *target* model (Qwen3-32B) greedily over a fixed prompt set in one of
two modes and grades the output:

* ``baseline``  — 32B prefills the prompt itself, then decodes.
* ``transfer``  — 32B loads the *source* (14B) cache for each prompt via
  :class:`~cross_model_kv.transfer_connector.CrossModelKVConnector`, applies the
  fitted mapper, injects the mapped KV, **skips its own prefill**, then decodes.

Both modes share identical prompt token ids (``--prompts-file``), so retention
``= transfer_acc / baseline_acc`` isolates the effect of substituting mapped
14B KV for the 32B's own prefill. Generated token ids are saved so a separate
step can also report greedy top-1 agreement (transfer vs baseline).

Example (ROCm)::

    VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN \\
    CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \\
    python scripts/eval_transfer.py --model Qwen/Qwen3-32B --mode transfer \\
        --prompts-file eval/prompts.json --gold-file eval/gold.json \\
        --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \\
        --source-cache-dir eval/14b --out eval/out_transfer.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_PKG_ROOT, os.environ.get("PYTHONPATH", "")) if p
)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str) -> str | None:
    """GSM8K few-shot answer: number after the *first* ``####`` (the current
    question's answer), else the first number on the answer line."""
    if "####" in text:
        tail = text.split("####", 1)[1]
        m = _NUM_RE.search(tail)
        if m:
            return m.group(0).replace(",", "").rstrip(".")
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def numbers_equal(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["baseline", "transfer"], required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--gold-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapper-path", default="")
    ap.add_argument("--source-cache-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    prompts: list[list[int]] = json.load(open(args.prompts_file))
    gold: list[str] = json.load(open(args.gold_file))
    assert len(prompts) == len(gold), "prompts/gold length mismatch"

    kv_cfg = None
    if args.mode == "transfer":
        if not args.mapper_path or not args.source_cache_dir:
            raise SystemExit("transfer mode needs --mapper-path and --source-cache-dir")
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
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max(args.max_model_len, 2048),
    )

    sp = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0,
        stop=["\nQuestion:", "\n\nQuestion:"],
    )
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp
    )

    results = []
    correct = 0
    for i, o in enumerate(outs):
        gen = o.outputs[0].text
        gen_ids = list(o.outputs[0].token_ids)
        pred = extract_answer(gen)
        ok = numbers_equal(pred, gold[i])
        correct += int(ok)
        results.append(
            {"idx": i, "pred": pred, "gold": gold[i], "correct": ok,
             "gen_ids": gen_ids, "gen": gen[:2000]}
        )

    acc = correct / len(prompts)
    summary = {"mode": args.mode, "model": args.model, "n": len(prompts),
               "correct": correct, "acc": acc}
    json.dump({"summary": summary, "results": results}, open(args.out, "w"))
    print(f"[eval:{args.mode}] acc = {acc:.4f} ({correct}/{len(prompts)}) -> {args.out}")


if __name__ == "__main__":
    main()
