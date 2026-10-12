# Reproducing Cross-Model KV Transfer Results

End-to-end recipe to reproduce every number in [`EVAL_STATUS.md`](EVAL_STATUS.md):
the fitted mapper R², the 5-benchmark accuracy retention, and the prefill-skip
latency sweep — for **Qwen3-14B → Qwen3-32B** on AMD ROCm (gfx950), TP=1.

Everything runs offline in three stages:

1. **Calibration** — capture matched KV from both models on a shared prompt set,
   then fit the per-head ridge mapper (`→ R²`).
2. **Accuracy eval** — GSM8K (generative CoT) + ARC/HellaSwag/WinoGrande/MMLU
   (generative letter-pick), baseline vs. transfer (`→ retention`).
3. **Performance eval** — single-request TTFT sweep, baseline vs. transfer
   (`→ speedup`).

> All commands are run from the package root
> `/home/hangy/vllm/cross_model_kv_cache_transfer`. A scratch data root
> `$DATA` holds captured caches and prompt/result JSON (it gets large — ~60 GB).

---

## 0. Environment

This is a **ROCm gfx950** box. Torch is the ROCm build (no CPU LAPACK), so the
ridge solve is offloaded to GPU (`--device cuda`), and `flash_attn` (a stray
CUDA-built wheel) is shadowed by a process-local shim so vLLM falls back to the
AITER unified-attention backend instead of crashing on import.

```bash
cd /home/hangy/vllm/cross_model_kv_cache_transfer

# Scratch root for captured KV + prompt/result JSON.
export DATA=/home/hangy/xmodel_kv
mkdir -p "$DATA"/{calib,eval,mc,lat}

# Process-local flash-attn shim (raises ModuleNotFoundError) + package on path.
export PYTHONPATH="$PWD/.fa_shim:$PWD:$PYTHONPATH"

# ROCm attention backend (flash-attn unavailable).
export VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN

# Two Qwen3 models, one per GPU, TP=1. Pin each command to a single device
# with CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES (shown per-step below).
```

Notes:
- **Python.** This ROCm image uses the system interpreter (`/usr/bin/python`);
  there is no project `.venv` here. Torch `2.11.0+git…`, vLLM `0.29.x`.
- **GPU memory.** The box is shared. `--gpu-memory-utilization` is a fraction of
  each GPU's *total* (~252 GB). Values below (0.4–0.5) leave room for a
  co-tenant; raise them if the GPU is idle.
- **`max_num_batched_tokens`.** `capture_kv.py` derives it from `--seq-len`, and
  every eval script needs `max_num_batched_tokens ≥ max_model_len`. For long
  prompts, pass a `--seq-len`/`--max-model-len` large enough to cover the
  longest prompt (called out in the latency step).

Sanity check the toolchain:

```bash
python -c "import torch, vllm; print('torch', torch.__version__, 'vllm', vllm.__version__)"
python -c "import flash_attn" 2>&1 | grep -q "cross_model_kv shim" && echo "flash-attn shim active"
```

---

## Stage 1 — Calibration (capture → fit → R²)

### 1a. Capture matched KV from both models

Both models must see the **identical** token ids so the per-token KV rows line
up. Capture 14B first while saving the shared prompt set, then replay the same
prompts through 32B. 64 sequences × 1024 tokens (FineWeb-style) is enough for a
stable fit.

```bash
# Source (14B) on GPU 0 — build + save the shared prompts.
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python scripts/capture_kv.py \
    --model Qwen/Qwen3-14B --tag 14b \
    --capture-dir "$DATA/calib" \
    --num-seqs 64 --seq-len 1024 \
    --save-prompts "$DATA/calib/prompts.json" \
    --max-model-len 2048 --gpu-memory-utilization 0.4

# Target (32B) on GPU 1 — replay the exact same prompts.
CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/capture_kv.py \
    --model Qwen/Qwen3-32B --tag 32b \
    --capture-dir "$DATA/calib" \
    --prompts-file "$DATA/calib/prompts.json" \
    --num-seqs 64 --seq-len 1024 \
    --max-model-len 2048 --gpu-memory-utilization 0.5
```

This writes `$DATA/calib/14b/` (40 layers) and `$DATA/calib/32b/` (64 layers) of
per-sequence KV shards, each row shaped `(seq_len, 2, n_kv=8, d_h=128)`. Models
download automatically on first use (`HF_HOME` cache).

### 1b. Fit the per-head ridge mapper

The fit runs on GPU (ROCm has no CPU LAPACK). `--top-k 8` selects the 8
best-correlated source layers per target layer; `--ridge-lambda 0.01` is the
regularizer. RoPE θ is 1e6 for both Qwen3 models (RoPE is stripped before the
solve and re-applied at inject time).

```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python scripts/fit_mapper.py \
    --source-dir "$DATA/calib/14b" \
    --target-dir "$DATA/calib/32b" \
    --out cross_model_kv/artifacts/qwen3_14b_to_32b \
    --top-k 8 --ridge-lambda 0.01 --device cuda \
    --source-model Qwen/Qwen3-14B --target-model Qwen/Qwen3-32B
```

The fit prints the **in-sample R²** (this is criterion #1) and the per-target
source-layer selection, then writes the artifact:

```
[fit] in-sample R^2  K=0.947  V=0.799     # target: K≈0.79, V≈0.65  → PASS
[fit] wrote mapper -> cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors
```

Outputs (~4.3 GB `.safetensors` + `.json` config). The source-layer map is a
clean monotonic diagonal (target *i* ← source ≈ 0.625·*i*, the 40→64 depth
ratio). **R² is only printed, not stored** — capture it from stdout, e.g.
`… | tee $DATA/calib/fit.log`.

---

## Stage 2 — Accuracy eval (retention)

Every eval loads the target model (Qwen3-32B) twice:

- **baseline** — 32B prefills the prompt itself.
- **transfer** — 32B loads the matching 14B cache, applies the mapper, injects
  the mapped KV, and **skips prefill**.

The connector matches a request to its source cache by a blake2b hash of the
full prompt token ids, so the transfer run needs a **14B capture of the exact
eval prompts** (not the calibration prompts). `retention = transfer_acc /
baseline_acc`.

### 2a. GSM8K (generative CoT, exact-match)

**Build 100 GSM8K 4-shot prompts** (gold answers are the post-`####` number):

```bash
python - "$DATA/eval" <<'PY'
import json, sys, re
from datasets import load_dataset
from transformers import AutoTokenizer
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
ds = load_dataset("openai/gsm8k", "main")
shots = ds["train"].select(range(4))
hdr = "\n\n".join(f"Question: {r['question']}\nAnswer: {r['answer']}" for r in shots)
prompts, gold = [], []
for r in ds["test"].select(range(100)):
    text = hdr + f"\n\nQuestion: {r['question']}\nAnswer:"
    prompts.append(tok(text).input_ids)
    gold.append(r["answer"].split("####")[-1].strip().replace(",", ""))
json.dump(prompts, open(f"{out}/prompts.json", "w"))
json.dump(gold,    open(f"{out}/gold.json", "w"))
print("gsm8k prompts:", len(prompts), "max len", max(len(p) for p in prompts))
PY
```

**Capture the 14B cache for those prompts** (source side of transfer):

```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python scripts/capture_kv.py \
    --model Qwen/Qwen3-14B --tag 14b \
    --capture-dir "$DATA/eval" \
    --prompts-file "$DATA/eval/prompts.json" \
    --num-seqs 100 --max-model-len 2048 --gpu-memory-utilization 0.4
```

**Run baseline then transfer** on 32B (GPU 1):

```bash
CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/eval_transfer.py --model Qwen/Qwen3-32B --mode baseline \
    --prompts-file "$DATA/eval/prompts.json" --gold-file "$DATA/eval/gold.json" \
    --max-model-len 2048 --gpu-memory-utilization 0.5 \
    --out "$DATA/eval/out_baseline.json"

CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/eval_transfer.py --model Qwen/Qwen3-32B --mode transfer \
    --prompts-file "$DATA/eval/prompts.json" --gold-file "$DATA/eval/gold.json" \
    --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \
    --source-cache-dir "$DATA/eval/14b" \
    --max-model-len 2048 --gpu-memory-utilization 0.5 \
    --out "$DATA/eval/out_transfer.json"
```

Each prints `[eval:MODE] acc = …`. Expected: **baseline 0.93, transfer 0.90 →
96.8% retention**.

### 2b. ARC-Challenge / HellaSwag / WinoGrande / MMLU (generative letter-pick)

These are scored **generatively** (few-shot, model emits an answer letter) — not
by log-likelihood `acc_norm` — because prefill-skip hides the continuation
logprobs. Baseline and transfer use the identical protocol, so the retention
ratio is apples-to-apples (absolute accuracy differs from the paper's LL
numbers; see [`EVAL_STATUS.md`](EVAL_STATUS.md)).

**Build 100 items/benchmark** (ARC/HS/WG 4-shot, MMLU 5-shot) → 400 prompts:

```bash
python scripts/build_mc_prompts.py --tokenizer Qwen/Qwen3-14B \
    --out-dir "$DATA/mc" --n 100
```

**Capture the 14B cache for all 400 MC prompts:**

```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python scripts/capture_kv.py \
    --model Qwen/Qwen3-14B --tag 14b \
    --capture-dir "$DATA/mc" \
    --prompts-file "$DATA/mc/prompts.json" \
    --num-seqs 400 --max-model-len 2048 --gpu-memory-utilization 0.4
```

**Run baseline then transfer** (per-benchmark accuracy grades against
`meta.json`):

```bash
CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/eval_mc.py --model Qwen/Qwen3-32B --mode baseline \
    --prompts-file "$DATA/mc/prompts.json" --meta-file "$DATA/mc/meta.json" \
    --max-model-len 2048 --gpu-memory-utilization 0.5 \
    --out "$DATA/mc/out_baseline.json"

CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/eval_mc.py --model Qwen/Qwen3-32B --mode transfer \
    --prompts-file "$DATA/mc/prompts.json" --meta-file "$DATA/mc/meta.json" \
    --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \
    --source-cache-dir "$DATA/mc/14b" \
    --max-model-len 2048 --gpu-memory-utilization 0.5 \
    --out "$DATA/mc/out_transfer.json"
```

### 2c. Compute retention table

```bash
python - "$DATA" <<'PY'
import json, sys
D = sys.argv[1]
gb = json.load(open(f"{D}/eval/out_baseline.json"))["summary"]["acc"]
gt = json.load(open(f"{D}/eval/out_transfer.json"))["summary"]["acc"]
mb = json.load(open(f"{D}/mc/out_baseline.json"))["summary"]
mt = json.load(open(f"{D}/mc/out_transfer.json"))["summary"]
rows = [("gsm8k", gb, gt)] + [
    (b, mb[b]["acc"], mt[b]["acc"]) for b in sorted(mb)
]
print(f"{'benchmark':>14}  {'base':>5} {'xfer':>5} {'ret%':>6}")
rets = []
for b, base, xfer in rows:
    r = 100 * xfer / base
    rets.append(r)
    print(f"{b:>14}  {base:5.3f} {xfer:5.3f} {r:6.2f}")
print(f"{'AVG':>14}  {'':5} {'':5} {sum(rets)/len(rets):6.2f}   (criterion >=95%)")
PY
```

Expected (matches [`EVAL_STATUS.md`](EVAL_STATUS.md)):

| Benchmark | Baseline | Transfer | Retention |
|---|---:|---:|---:|
| GSM8K | 0.930 | 0.900 | 96.8% |
| ARC-Challenge | 0.950 | 0.910 | 95.8% |
| HellaSwag | 0.780 | 0.700 | 89.7% |
| MMLU | 0.700 | 0.710 | 101.4% |
| WinoGrande | 0.760 | 0.770 | 101.3% |
| **Average** | | | **97.0%** ✅ |

**Injection is real, not a silent fallback.** Under greedy decode a fallback to
normal prefill would produce byte-identical output; the transfer runs diverge
(99/100 GSM8K generations differ), which is only possible if mapped-KV injection
drove the model. To verify yourself:

```bash
python - "$DATA/eval" <<'PY'
import json, sys
D = sys.argv[1]
b = json.load(open(f"{D}/out_baseline.json"))["results"]
t = json.load(open(f"{D}/out_transfer.json"))["results"]
diff = sum(1 for x, y in zip(b, t) if x["gen_ids"] != y["gen_ids"])
print(f"{diff}/{len(b)} transfer generations differ from baseline")
PY
```

---

## Stage 3 — Performance eval (prefill-skip latency)

Single-request TTFT on 32B across prompt lengths, baseline vs. transfer, with
prefix caching off. Longest prompt is 16384 tokens, so `max_model_len` and the
capture `--seq-len` must both be ≥ ~20000.

**Build one prompt per length** (512 → 16384):

```bash
python - "$DATA/lat" <<'PY'
import json, sys
from datasets import load_dataset
from transformers import AutoTokenizer
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
lens = [512, 1024, 2048, 4096, 8192, 16384]
# Concatenate FineWeb-ish text, then trim to each exact length.
ds = load_dataset("openai/gsm8k", "main")["train"]
big = tok("\n\n".join(r["question"] + " " + r["answer"] for r in ds.select(range(4000)))).input_ids
prompts = [big[:L] for L in lens]
json.dump(prompts, open(f"{out}/prompts.json", "w"))
json.dump(lens,    open(f"{out}/lens.json", "w"))
print("lat lens:", lens, "have", [len(p) for p in prompts])
PY
```

**Capture the 14B cache** (`--seq-len 20000` lifts `max_num_batched_tokens`
above the 16k prompt):

```bash
CUDA_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 \
python scripts/capture_kv.py \
    --model Qwen/Qwen3-14B --tag 14b \
    --capture-dir "$DATA/lat" \
    --prompts-file "$DATA/lat/prompts.json" \
    --num-seqs 6 --seq-len 20000 \
    --max-model-len 20000 --gpu-memory-utilization 0.45
```

**Run the sweep** (warmup primes the page cache so timed transfer iterations
measure compute + H2D, not cold disk):

```bash
CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/bench_latency.py --model Qwen/Qwen3-32B --mode baseline \
    --prompts-file "$DATA/lat/prompts.json" --lens-file "$DATA/lat/lens.json" \
    --max-model-len 20000 --gpu-memory-utilization 0.5 \
    --iters 3 --warmup 1 --out "$DATA/lat/ttft_baseline.json"

CUDA_VISIBLE_DEVICES=1 HIP_VISIBLE_DEVICES=1 \
python scripts/bench_latency.py --model Qwen/Qwen3-32B --mode transfer \
    --prompts-file "$DATA/lat/prompts.json" --lens-file "$DATA/lat/lens.json" \
    --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \
    --source-cache-dir "$DATA/lat/14b" \
    --max-model-len 20000 --gpu-memory-utilization 0.5 \
    --iters 3 --warmup 1 --out "$DATA/lat/ttft_transfer.json"
```

**Compute speedups:**

```bash
python - "$DATA/lat" <<'PY'
import json, sys
D = sys.argv[1]
b = {r["len"]: r["median_s"] for r in json.load(open(f"{D}/ttft_baseline.json"))["rows"]}
t = {r["len"]: r["median_s"] for r in json.load(open(f"{D}/ttft_transfer.json"))["rows"]}
print(f"{'tokens':>7} {'base ms':>9} {'xfer ms':>9} {'speedup':>8}")
for L in sorted(b):
    print(f"{L:>7} {b[L]*1e3:9.1f} {t[L]*1e3:9.1f} {b[L]/t[L]:7.2f}x")
PY
```

Expected trend (crossover ~2k tokens, gap widens monotonically):

| Prompt tokens | Baseline | Transfer | Speedup |
|---:|---:|---:|---:|
| 512   |   51.8 ms |   89.1 ms | 0.58× |
| 2048  |  160.3 ms |  143.8 ms | 1.11× |
| 8192  |  685.4 ms |  376.6 ms | 1.82× |
| 16384 | 1614.1 ms |  690.5 ms | 2.34× |

The DESIGN target (>3× at ≥8k) is **not yet reached** in this eager-mode MVP —
see the "Why below the paper" and "Next steps" notes in
[`EVAL_STATUS.md`](EVAL_STATUS.md).

---

## Stage 4 — Live GPU-to-GPU transfer (production path)

Stages 1–3 stage KV through disk (source writes `.safetensors`, target reads them back
host-side). The production path runs **both models concurrently** and moves each
sequence's KV **GPU-to-GPU** over RCCL/NCCL — no disk, no host copy. See
[`REPORT.md`](REPORT.md) §7 for the design.

### 4a. Transport unit test (byte-exact, no models)

```bash
export PYTHONPATH="$PWD/.fa_shim:$PWD:$PYTHONPATH"
python -m pytest tests/test_kv_link.py -v
```

Spawns a rank-0/rank-1 pair on `cuda:0`/`cuda:1`, sends a known `[L, N, H, D]` KV pair,
and asserts the receiver reconstructs it byte-exactly (`torch.equal`) with the matching
`seq_hash`. Requires 2 visible GPUs (skips otherwise).

### 4b. Live 14B→32B end-to-end

The driver re-execs itself once per role with `CUDA_VISIBLE_DEVICES` pinned, so each vLLM
engine owns one GPU. Producer (14B) = rank 0 launches the store; consumer (32B) = rank 1.

```bash
export PYTHONPATH="$PWD/.fa_shim:$PWD:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN
python scripts/run_live_transfer.py \
    --source-model Qwen/Qwen3-14B --target-model Qwen/Qwen3-32B \
    --mapper-path cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors \
    --smoke-lengths 256 --with-baseline \
    --producer-gpu 0 --consumer-gpu 1 \
    --max-tokens 8 --max-model-len 1024 --gpu-memory-utilization 0.5 \
    --out /tmp/live_out.json
```

Expected on success (needs ~62 GB free on the 32B GPU + ~29 GB on the 14B GPU):

```text
cross-model KV hit: skipping prefill of 192 tokens for req ...
sent source KV over NCCL: 40 layers, 256 tokens (req hash 65d92694)
[producer] prefilled + sent 1 sequences in ...s
[orch] live transfer complete -> /tmp/live_out.json
```

Notes:

- **Prompt length must span ≥ 2 blocks.** `block_size = 64` under
  `ROCM_AITER_UNIFIED_ATTN`, so `serve_matched(N, 64) = ((N-1)//64)*64` is **0** for
  N ≤ 64 — a single-block prompt has nothing to transfer (the target must run it to
  produce logits). Use `--smoke-lengths 256` (or any N ≥ 128), not 64.
- The `--with-baseline` divergence check is a *weak* control on deterministic prompts:
  faithful injection reproduces the same greedy tokens as native prefill, so 0 divergence
  is expected and not a failure. The definitive positive control is the
  `cross-model KV hit` / `sent source KV over NCCL` log pair above (proof captured in
  `logs/live_smoke3_proof.txt`).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Invalid HF URI` on GSM8K | Use `openai/gsm8k` (not `gsm8k`). |
| `max_num_batched_tokens < max_model_len` | `capture_kv.py` sets it from `--seq-len`; raise `--seq-len` (and `--max-model-len`) above the longest prompt. |
| OOM: `Free memory … < desired …` | `--gpu-memory-utilization` is a fraction of the GPU's *total* (~252 GB) shared with co-tenants — lower it (0.4/0.5). |
| GSM8K baseline ≈ 0.18 | Model runs on into extra hallucinated Q/A; already fixed — `eval_transfer.extract_answer` takes the **first** `####` and `stop=["\nQuestion:"]`. |
| No `cross-model KV hit` log lines | The connector loggers are now parented under `vllm.*` so their INFO lines show in the EngineCore subprocess. If still absent, the prompt is too short (single block → `serve_matched = 0`; use N ≥ 128) or injection did not run — confirm via generation divergence (Stage 2c). |
| `ImportError: flash_attn_2_cuda` | Ensure `.fa_shim` is first on `PYTHONPATH` so the shim raises `ModuleNotFoundError` and vLLM falls back to AITER. |

## What each script does

| Script | Role |
|---|---|
| `scripts/capture_kv.py` | Run a model over prompts, dump per-sequence KV shards (`--save-prompts` / `--prompts-file` to share the token ids). |
| `scripts/fit_mapper.py` | Offline per-head ridge fit (top-k source selection, RoPE strip/apply) → mapper artifact + printed R². |
| `scripts/eval_transfer.py` | GSM8K generative exact-match, baseline vs. transfer. |
| `scripts/build_mc_prompts.py` | Build combined ARC/HellaSwag/WinoGrande/MMLU few-shot letter-pick prompts + `meta.json`. |
| `scripts/eval_mc.py` | MC letter-pick accuracy per benchmark, baseline vs. transfer. |
| `scripts/bench_latency.py` | TTFT sweep, baseline vs. transfer. |
| `scripts/run_live_transfer.py` | Live GPU-to-GPU 14B→32B driver: concurrent producer/consumer, KV over RCCL/NCCL (Stage 4). |
| `cross_model_kv/kv_link.py` | 2-rank NCCL point-to-point KV channel (`StatelessProcessGroup` header + `PyNcclCommunicator` data plane). |
| `cross_model_kv/transfer_producer_connector.py` | Serving producer (`CrossModelKVSender`): sends whole-sequence KV over the link instead of writing disk. |
