# Cross-Model KV Transfer — Evaluation Status

Real end-to-end evaluation of the closed-form per-head ridge KV mapper
(**Qwen3-14B → Qwen3-32B**, AMD ROCm / gfx950, TP=1 on GPUs 0–1) against the
three DESIGN success criteria. All numbers below are from live vLLM runs on the
downloaded models, not simulation.

## Setup

| Item | Value |
|---|---|
| Source / target | `Qwen/Qwen3-14B` (40L) → `Qwen/Qwen3-32B` (64L), shared tokenizer |
| Matched KV geometry | `n_kv=8`, `d_h=128`; top-k=8 cross-layer, feature_dim=8192 |
| Attention backend | `ROCM_AITER_UNIFIED_ATTN` (flash-attn shimmed off on ROCm) |
| Transport | offline: shared filesystem, keyed by blake2b hash of full prompt token ids; live: **GPU-to-GPU** RCCL/NCCL peer DMA (`kv_link.py`), byte-exact (`tests/test_kv_link.py` passed), 14B→32B smoke-run confirmed |
| Calibration | 64 FineWeb sequences × 1024 tokens, identical for both models |
| Mapper artifact | `cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors` (4.3 GB) |

## Results vs. success criteria

### 1. In-sample fit R² — ✅ PASS (exceeds target)

| | K | V |
|---|---|---|
| **Measured** | **0.947** | **0.799** |
| DESIGN target (paper, k=8) | ≈0.79 | ≈0.65 |

Source-layer selection is a clean monotonic diagonal (target layer *i* ← source
layers ≈ 0.625·*i*, matching the 40→64 depth ratio).

### 2. Accuracy retention — ✅ PASS (avg 97.0% across 5 benchmarks)

Real Qwen3-32B, 100 items per benchmark, greedy decode, `retention =
transfer_acc / baseline_acc`.

| Benchmark | Baseline | Transfer | Retention |
|---|---:|---:|---:|
| GSM8K (4-shot CoT, exact-match) | 0.930 | 0.900 | 96.8% |
| ARC-Challenge (letter-pick) | 0.950 | 0.910 | 95.8% |
| HellaSwag (letter-pick) | 0.780 | 0.700 | 89.7% |
| WinoGrande (letter-pick) | 0.760 | 0.770 | 101.3% |
| MMLU (5-shot, letter-pick) | 0.700 | 0.710 | 101.4% |
| **Average** | | | **97.0%** (criterion ≥95%; paper 97.6%) |

**Scoring protocol.** GSM8K is generative exact-match. The four MC benchmarks are
scored **generatively** — few-shot, the model emits an answer letter — *not* by
per-token log-likelihood (`acc_norm`). This is because the transfer connector
*skips prefill* for the injected prefix, so continuation logprobs inside that
region aren't exposed. Baseline and transfer use the identical protocol, so the
retention *ratio* is apples-to-apples; the absolute accuracies are not directly
comparable to the paper's log-likelihood numbers. HellaSwag is the weakest pair
(89.7%), consistent with letter-pick being more sensitive to KV fidelity than LL.

**Mechanism confirmed:** on GSM8K, 99/100 transfer generations differ from
baseline (greedy token agreement 3.5%). A silent fallback to normal prefill would
produce byte-identical greedy output, so the mapped-KV injection provably drove
the target model. The MC runs (identical connector code path) also diverge — 70
items across the four benchmarks — confirming injection there too; per-benchmark
divergence is low only because a single high-confidence answer letter usually
survives the KV substitution.

### 3. Prefill-skip latency — ⚠️ PARTIAL (clear speedup, grows with length; 3× not yet reached at 8k)

Single-request TTFT (median of 3, prefix caching off, page cache warmed).

| Prompt tokens | Baseline TTFT | Transfer TTFT | Speedup |
|---:|---:|---:|---:|
| 512   |   51.8 ms |   89.1 ms | 0.58× |
| 1024  |   87.3 ms |  106.7 ms | 0.82× |
| 2048  |  160.3 ms |  143.8 ms | 1.11× |
| 4096  |  322.0 ms |  222.2 ms | 1.45× |
| 8192  |  685.4 ms |  376.6 ms | 1.82× |
| 16384 | 1614.1 ms |  690.5 ms | **2.34×** |

Baseline prefill grows ~superlinearly (O(N²) attention); transfer grows ~linearly
(per-token mapping + scatter). Crossover is at ~2k tokens and the gap widens
monotonically. The DESIGN criterion (>3× at ≥8k) is **not yet met** at these
lengths in this MVP: we reach 1.8× at 8k and 2.3× at 16k.

Why below the paper's 4–25×:
- Mapping runs in **eager mode** (`enforce_eager=True`), a Python loop over 64
  target layers — no CUDA-graph capture or kernel fusion.
- Per-request the source cache is re-loaded and copied host→device each iter
  (filesystem MVP transport), a fixed overhead visible at short lengths
  (transfer is *slower* than baseline below ~2k tokens).
- Paper speedups extend to 32k tokens, where the O(N²) baseline gap is larger.

## Artifacts

| Path | Contents |
|---|---|
| `cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors` + `.json` | fitted mapper + config |
| `scripts/fit_mapper.py` | offline fit → R² report |
| `scripts/eval_transfer.py` | GSM8K accuracy, baseline vs transfer |
| `scripts/build_mc_prompts.py` | build ARC/HellaSwag/WinoGrande/MMLU generative prompts |
| `scripts/eval_mc.py` | MC letter-pick accuracy, baseline vs transfer |
| `scripts/bench_latency.py` | TTFT sweep, baseline vs transfer |
| `~/xmodel_kv/eval/accuracy_summary.json` | GSM8K retention |
| `~/xmodel_kv/mc/retention_full.json` | 5-benchmark retention table |
| `~/xmodel_kv/lat/ttft_{baseline,transfer}.json` | raw latency samples |

## Next steps

1. **Close the latency gap to >3×.** Highest-leverage items:
   - Capture the mapping loop under a CUDA graph / compile it (drop eager).
   - Batch the per-layer ridge apply into fewer large matmuls. The per-request H2D
     copy is now removed by the live GPU-to-GPU transport (`kv_link.py`, see
     REPORT.md §7); re-running the §4.2 sweep over that path to quantify the gain is
     the remaining measurement.
   - Extend the sweep to 32k tokens to characterize the asymptote vs. the paper.
2. **Match the paper's log-likelihood `acc_norm`** for the MC benchmarks (current
   numbers are generative letter-pick). This needs the connector to inject only
   the shared *context* prefix and forward each continuation so its logprobs are
   scored — i.e. prefix-hash matching in the connector, not full-prompt matching.
3. **k-ablation** (k=1→8) and RoPE-stripped vs. coupled, to reproduce the
   paper's ablations and confirm k=8 is the right operating point.
4. **Two-process disaggregated demo** (14B producer + 32B consumer live) instead
   of offline capture, per DESIGN M4.
