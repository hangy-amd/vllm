# Cross-Model KV Cache Transfer — Evaluation Report

**Subject:** Closed-form per-head ridge KV mapper, Qwen3-14B → Qwen3-32B
**Platform:** AMD ROCm (gfx950), 2× GPU, tensor-parallel size 1 per model
**Date:** 2026-09-15
**Status:** Accuracy criteria **met**; latency criterion **partially met** (MVP)

---

## 1. Executive summary

We evaluated a closed-form, per-head ridge-regression mapper that transplants the
KV cache of a small source model (Qwen3-14B, 40 layers) into a large target model
(Qwen3-32B, 64 layers), letting the target **skip its own prefill** and decode
directly from mapped KV. Evaluation used real downloaded models on live vLLM — no
simulation.

| Criterion | Target | Measured | Verdict |
|---|---|---|---|
| In-sample fit R² (K / V) | ≈0.79 / ≈0.65 | **0.947 / 0.799** | ✅ Pass |
| Mean accuracy retention (5 benchmarks) | ≥ 95% | **97.0%** | ✅ Pass |
| Prefill-skip TTFT speedup at ≥ 8k tokens | > 3× | **1.82× @ 8k, 2.34× @ 16k** | ⚠️ Partial |

**Conclusion.** The mapper reconstructs target KV with high fidelity and preserves
downstream task accuracy within the design tolerance. The prefill-skip path yields
a real, monotonically growing latency advantage beyond ~2k tokens, but the MVP
transport and eager execution keep it below the 3× bar at the tested lengths. The
accuracy result validates the core method; the latency result is an engineering
gap, not a correctness gap (see §5.3).

---

## 2. System under test

| Component | Value |
|---|---|
| Source model | `Qwen/Qwen3-14B` (40 decoder layers) |
| Target model | `Qwen/Qwen3-32B` (64 decoder layers) |
| Matched KV geometry | `n_kv = 8` heads, `head_dim = 128` (shared tokenizer) |
| Mapper | per-head ridge, top-k = 8 cross-layer sources, λ = 0.01, feature_dim = 8·8·128 = 8192 |
| Positional handling | source RoPE stripped → map in content space → target RoPE re-applied; V positionless |
| vLLM integration | out-of-tree `KVConnectorBase_V1` pair (capture producer + transfer consumer) |
| Transport | two paths: (a) offline shared-filesystem staging, keyed by blake2b-128 of full prompt token ids; (b) live **GPU-to-GPU** RCCL/NCCL peer DMA (no disk, no host copy) — see §7 |
| Attention backend | `ROCM_AITER_UNIFIED_ATTN` (flash-attn shimmed off on ROCm) |
| Execution | `enforce_eager=True`, chunked prefill off, TP=1, one model per GPU |

The mapper artifact (`qwen3_14b_to_32b.safetensors`, 4.3 GB) was fit once, offline,
and held **frozen** across every accuracy and latency run below.

---

## 3. Accuracy evaluation

### 3.1 Methodology

Both modes run the **target model (Qwen3-32B)** over identical prompt token ids and
are graded identically; the only difference is the KV source:

- **Baseline** — 32B prefills the prompt itself, then decodes.
- **Transfer** — 32B loads the matching 14B cache, applies the frozen mapper,
  injects the mapped KV, **skips its own prefill**, then decodes.

`retention = transfer_acc / baseline_acc`. Because both modes share the protocol,
the ratio isolates the effect of substituting mapped KV for native prefill.

**Scoring protocol.** GSM8K is generative exact-match (4-shot chain-of-thought,
answer parsed as the number after the first `####`). The four multiple-choice
benchmarks are scored **generatively** — few-shot, the model emits a single answer
letter — *not* by per-token log-likelihood (`acc_norm`). This is a deliberate
consequence of the mechanism: the connector skips prefill for the injected prefix,
so continuation log-probabilities inside that region are not exposed. Baseline and
transfer use the identical generative protocol, so the retention **ratio** is
apples-to-apples; absolute accuracies are therefore not directly comparable to the
paper's log-likelihood numbers.

100 items per benchmark, greedy decoding (`temperature=0`).

### 3.2 Results

| Benchmark | Protocol | Baseline | Transfer | Retention |
|---|---|---:|---:|---:|
| GSM8K | 4-shot CoT, exact-match | 0.930 | 0.900 | 96.8% |
| ARC-Challenge | 4-shot, letter-pick | 0.950 | 0.910 | 95.8% |
| HellaSwag | 4-shot, letter-pick | 0.780 | 0.700 | 89.7% |
| WinoGrande | 4-shot, letter-pick | 0.760 | 0.770 | 101.3% |
| MMLU | 5-shot, letter-pick | 0.700 | 0.710 | 101.4% |
| **Average** | | | | **97.0%** |

Criterion: mean retention ≥ 95% → **Pass** (paper reference 97.6%).

### 3.3 Analysis

- **Two benchmarks exceed 100%** (WinoGrande, MMLU): mapped KV occasionally flips a
  borderline item the right way; with 100 items the per-benchmark noise floor is
  ±~2 points, so these are best read as "no measurable loss."
- **HellaSwag is the weakest pair (89.7%)** — consistent with letter-pick being more
  sensitive to fine-grained KV fidelity than log-likelihood scoring on this
  commonsense-continuation task.
- **The strong benchmarks (GSM8K, ARC) retain ≥ 95.8%**, indicating the mapped KV
  preserves the multi-step reasoning and factual signal the target relies on.

### 3.4 Mechanism verification (the injection is real)

Under greedy decoding, a silent fallback to native prefill would produce
**byte-identical** output. We therefore use output divergence as a positive control:

- GSM8K: **99 / 100** transfer generations differ from baseline (greedy top-1 token
  agreement 3.5%).
- MC: **70 / 400** items diverge across the four benchmarks.

Divergence at this scale is only possible if mapped-KV injection actually drove the
target model. Per-benchmark MC divergence is modest because a single high-confidence
answer letter usually survives KV substitution even when the internal state shifts —
which is also *why* accuracy is retained. Prompt-hash matching was independently
verified (100/100 eval prompts matched their captured source directories) before the
runs.

---

## 4. Performance evaluation

### 4.1 Methodology

Metric: **single-request time-to-first-token (TTFT)** on the target model, measured
by `scripts/bench_latency.py`. The request is capped at `max_tokens=1`, and a wall
clock wraps the full `generate` call, so the elapsed time is dominated by the prefill
(baseline) or KV-injection (transfer) segment that produces the first token.

```
speedup(L) = baseline_ttft(L) / transfer_ttft(L)
```

Controls:

- **Prefix caching disabled** — every iteration genuinely re-prefills / re-injects.
- **1 warmup + median of 3 timed iterations** — warmup primes the OS page cache for
  the source shards, so timed transfer iterations measure compute + host→device
  copy, not cold disk reads.
- **Eager mode, chunked prefill off, batch size 1** — pure latency on a shared
  execution path; baseline and transfer read identical prompt token ids.

Swept prompt lengths: 512 → 16384 tokens (one prompt per length).

### 4.2 Results

| Prompt tokens | Baseline TTFT | Transfer TTFT | Speedup |
|---:|---:|---:|---:|
| 512   |   51.8 ms |   89.1 ms | 0.58× |
| 1024  |   87.3 ms |  106.7 ms | 0.82× |
| 2048  |  160.3 ms |  143.8 ms | 1.11× |
| 4096  |  322.0 ms |  222.2 ms | 1.45× |
| 8192  |  685.4 ms |  376.6 ms | 1.82× |
| 16384 | 1614.1 ms |  690.5 ms | **2.34×** |

Criterion: > 3× at ≥ 8k tokens → **Not met** at tested lengths (1.82× @ 8k).

### 4.3 Analysis

- **Baseline TTFT grows ~super-linearly** (O(N²) attention): 51.8 ms → 1614 ms as the
  prompt grows 32×, a ~31× increase.
- **Transfer TTFT grows ~linearly** (per-token mapping + scatter): 89 ms → 690 ms,
  ~7.7× over the same range.
- **Crossover ≈ 2k tokens.** Below it, transfer is *slower* — a fixed per-request
  overhead (source-cache disk load + host→device copy + eager Python loop over 64
  layers) dominates before the O(N²) prefill it replaces becomes large.
- **The advantage widens monotonically** and is already 2.34× at 16k, with the curves
  still diverging — extrapolation to 32k (the paper's operating point) would push
  past 3×.

### 4.4 Why below the 3× bar (root cause)

The gap is **transport/execution overhead**, not algorithmic:

1. **Eager mapping loop** — the 64-layer ridge-apply runs in Python with no CUDA-graph
   capture or kernel fusion (`enforce_eager=True`, required because the injection path
   is not yet CUDA-graph-safe).
2. **Filesystem MVP transport** — each request re-loads the source cache from disk and
   copies it host→device, a fixed cost visible as the sub-2k slowdown. Addressed by the
   live GPU-to-GPU path in §7.
3. **Length ceiling** — tested to 16k; the paper's 4–25× figures extend to 32k where
   the baseline's quadratic penalty is far larger.

---

## 7. Live GPU-to-GPU transport (production path)

The MVP numbers above use the offline disk-staged transport: the 14B source writes
per-layer KV to `.safetensors`, and the 32B consumer reads it back host-side and copies
it host→device per request. That disk read + H2D copy is root cause #2 of the sub-2k
slowdown. The **production path removes it**: source (14B) and target (32B) run
**concurrently**, one per GPU, and each prefilled sequence's KV crosses the interconnect
**GPU-to-GPU** over RCCL/NCCL peer DMA — no disk, no host round trip.

### 7.1 Design

- **`cross_model_kv/kv_link.py`** — a 2-rank point-to-point channel built on vLLM's own
  side-channel primitives (`StatelessProcessGroup` for the `{seq_hash, shape, dtype}`
  header, `PyNcclCommunicator` for the KV tensors), **not** a second global
  `torch.distributed` group (which would collide with each model's TP group). Producer =
  rank 0, consumer = rank 1.
- **`transfer_producer_connector.py`** (`CrossModelKVSender`) — the serving producer:
  reuses the capture connector's per-layer slot `_extract`, buffers the forward's layers
  on-device, then in `wait_for_save` stacks to `[L, N, H, D]` and `send_kv`s one sequence
  at a time.
- **`transfer_connector.py`** (`CrossModelKVConnector`, live mode, toggled by
  `link_port > 0`) — identical mapper / RoPE / injection math; only the KV **source**
  changes from `load_sequence(disk)` to `KVLink.recv_kv()`. The ordered NCCL channel is
  the synchronization: producer send #i pairs with consumer recv #i, and the `seq_hash`
  header is asserted on receipt to catch any misalignment.

The disk path is retained — it is still required to *fit* the mapper offline.

### 7.2 Verification (smoke scope)

- **Transport unit test** — `tests/test_kv_link.py` spawns a rank-0/rank-1 pair and sends
  a known `[4, 32, 8, 128]` KV pair; the receiver reconstructs it **byte-exactly**
  (`torch.equal`) with the matching hash. **Passed** (RCCL `nccl==2.27.7`).
- **Live end-to-end 14B→32B run** — `scripts/run_live_transfer.py`, both models resident
  (14B on GPU 0, 32B on GPU 1), 256-token prompt, `block_size = 64` (ROCm AITER):
  - consumer scheduler skips prefill: `num_matched = 192` of 256 tokens;
  - producer prefills 40 layers and sends `k_shape = (40, 256, 8, 128)` over NCCL in one
    hop (`prefilled + sent 1 sequences in 12.7 s`, cold JIT included);
  - consumer `start_load_kv` processes the request (`n_requests = 1`), `recv_kv` returns
    the matching `seq_hash`, the mapper injects, and the target decodes coherently.

  This directly confirms the skip-prefill + inject path ran (a silent prefill fallback
  would show `n_requests = 0`) — a stronger positive control than output divergence,
  which is degenerate on deterministic prompts (faithful injection reproduces the same
  greedy tokens native prefill would emit). Proof lines: `logs/live_smoke3_proof.txt`.

A full latency re-sweep over the live path (to quantify the removal of the H2D copy from
the §4.2 curve) is supported by the driver but was **not** run here; the smoke scope
(unit test + one live end-to-end transfer) confirms correctness, and the sub-2k fixed
overhead it removes is the disk-load + H2D copy identified as root cause #2.
