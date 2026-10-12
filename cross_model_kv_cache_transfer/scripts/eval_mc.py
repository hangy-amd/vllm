"""Generative multiple-choice eval for cross-model KV transfer (M5).

Runs Qwen3-32B over the combined MC prompt set (``build_mc_prompts.py``) in
``baseline`` or ``transfer`` mode, greedily emits an answer letter per item, and
grades against ``meta.json``. Per-benchmark accuracy is written out; a companion
step computes retention = transfer_acc / baseline_acc. See ``build_mc_prompts``
for why MC is scored generatively rather than by log-likelihood.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
os.environ["PYTHONPATH"] = os.pathsep.join(
    p for p in (_PKG_ROOT, os.environ.get("PYTHONPATH", "")) if p
)

_LETTER_RE = re.compile(r"[A-E]")


def extract_letter(text: str) -> str | None:
    m = _LETTER_RE.search(text)
    return m.group(0) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["baseline", "transfer"], required=True)
    ap.add_argument("--prompts-file", required=True)
    ap.add_argument("--meta-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mapper-path", default="")
    ap.add_argument("--source-cache-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.inputs import TokensPrompt

    prompts: list[list[int]] = json.load(open(args.prompts_file))
    meta: list[dict] = json.load(open(args.meta_file))
    assert len(prompts) == len(meta)

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
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=max(args.max_model_len, 2048),
    )

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0,
                        stop=["\n", "\n\n", "Question:"])
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in prompts], sp, use_tqdm=True
    )

    per_bench = defaultdict(lambda: [0, 0])  # benchmark -> [correct, total]
    results = []
    for i, o in enumerate(outs):
        gen = o.outputs[0].text
        pred = extract_letter(gen)
        gold = meta[i]["gold"]
        ok = pred == gold
        b = meta[i]["benchmark"]
        per_bench[b][0] += int(ok)
        per_bench[b][1] += 1
        results.append({"idx": i, "benchmark": b, "pred": pred, "gold": gold,
                        "correct": ok, "gen_ids": list(o.outputs[0].token_ids),
                        "gen": gen[:200]})

    summary = {b: {"correct": c, "total": t, "acc": c / t}
               for b, (c, t) in sorted(per_bench.items())}
    json.dump({"mode": args.mode, "model": args.model,
               "summary": summary, "results": results}, open(args.out, "w"))
    print(f"[eval_mc:{args.mode}]")
    for b, s in summary.items():
        print(f"  {b:>14}: acc={s['acc']:.4f} ({s['correct']}/{s['total']})")


if __name__ == "__main__":
    main()
