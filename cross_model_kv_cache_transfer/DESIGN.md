# Cross-Model KV Cache Transfer in vLLM — Design & Implementation Plan

**Target pair:** Qwen3-14B → Qwen3-32B (small-to-large, "matched-KV").
**Paper:** Heo et al., *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form
Linear Mapping for Prefill Reuse* (arXiv:2608.03893).
**Reference code:** `algorithm.py` (ridge mapper) and `multi_gpu_comm.py` (P2P transport)
in this directory.
**Deployment target:** **AMD ROCm** — this environment is 8× **gfx950 (CDNA4, MI350-class)**,
ROCm/HIP 7.2, PyTorch 2.10+HIP. The paper's numbers are on 8×H100; all ROCm-specific
adaptations are called out in [§4.5](#45-rocm-specifics-must-read) and flagged inline.

---

## 1. Goal

When a serving stack swaps between two members of the same model family, the receiver
normally re-runs **prefill** to populate its KV cache. The paper shows the cross-model
KV relationship is largely **linear**, so the receiver can instead reuse the sender's
already-prefilled cache through a cheap, per-head, closed-form linear map — skipping
prefill entirely.

Concretely, we want two deliverables in vLLM:

1. **Offline calibration** that fits the ridge mapper for the Qwen3 14B→32B pair from a
   small FineWeb-Edu calibration set and serializes it to disk.
2. **Serving-time transfer**: the 32B receiver decodes directly from a mapped 14B cache,
   skipping its own prefill, and we measure (a) accuracy retention on standard benchmarks
   and (b) prefill-latency speedup.

The paper's headline for this exact pair: **97.6% average accuracy retention** across five
benchmarks at `k=8`, and **4–25× lower prefill latency** than re-prefill over 64–32k tokens.
This is the strongest pair in the paper — the right choice for a first implementation because
any regression is most visible.

---

## 2. Algorithm recap (what we must implement)

The mapper produces the target's per-head keys and values from the source's cache. It has
three components (paper §3; faithfully mirrored in `algorithm.py`):

### 2.1 Per-head ridge regression (§3.1)
For each target `(layer l, head h)` and each of K / V independently, solve the closed-form
ridge system in float32:

```
W* = (XᵀX + λI)⁻¹ XᵀY,   b = mean(Y) − mean(X) W*      (λ = 0.01)
```

- `X ∈ R^{N × (k·n_kv·d_h)}` — source features (below); `Y ∈ R^{N × d_h}` — target head.
- Center X and Y before solving; recover bias from means (`algorithm.py:ridge_fit`).
- Nothing is shared across heads or between K and V.

### 2.2 Cross-layer source selection (§3.2) — *largest quality contributor*
Each **target** layer draws from its **top-k most predictive source layers**. Rank source
layers by head-averaged single-source `R²` (averaged over K and V so K and V share one
selection), then concatenate all KV heads of the selected source layers as the regression
input `X` (`algorithm.py:_select_sources`, `_gather`). For 14B→32B the paper selects **k=8**.

### 2.3 Content-space mapping / RoPE factoring (§3.3)
Keys are stored **post-RoPE**. Before fitting we **strip RoPE** (`R_Θ⁻¹`, exact since RoPE
is orthogonal) so the map lives in a position-free content space and transfers across context
lengths. At inference: strip source RoPE → apply `W_K` → re-apply **target** RoPE.
Values carry no positional encoding and are mapped directly.

```
K̂_t = ( K_s · R_Θs⁻¹(t) · W_K + b_K ) · R_Θt(t)
V̂_t =   V_s · W_V + b_V
```

### 2.4 Calibration protocol
- Data: **500 FineWeb-Edu sequences × 1024 tokens**, stride-4 subsampled → ~128k
  token-level observations per target head.
- Precision: bf16 forward, **float32 covariance** (`XᵀX`), float32 solve.
- Fit cost in paper: ~52 min for this pair on one 8×H100 node, no gradients.
- Mapper size for 14B→32B at k=8: **~1.07 B params, ~4 GB** on disk. Not
  sequence-length dependent; can live on CPU/disk and page in.

---

## 3. Why Qwen3 14B→32B fits the "matched-KV" precondition

Confirmed from `vllm/model_executor/models/qwen3.py` and Qwen3 configs:

| Axis | Qwen3-14B (source) | Qwen3-32B (target) | Matched? |
|---|---|---|---|
| KV heads `n_kv` | 8 | 8 | ✅ |
| Head dim `d_h` | 128 | 128 | ✅ |
| Layers `L` | 40 | 64 | differ (ok) |
| Hidden size | 5120 | 5120 | — |
| Q heads | 40 | 64 | differ (ok) |
| RoPE `theta` | 1,000,000 | 1,000,000 | same here |

`n_kv` and `d_h` match — the precondition for the linear-structure result. Depth differs
(40→64), which is exactly what cross-layer top-k selection handles.

### 3.1 Critical subtlety: Qwen3 `q_norm`/`k_norm` before RoPE
Qwen3 applies **per-head `RMSNorm(head_dim)` to K (and Q) *before* RoPE**
(`qwen3.py:150-151`, applied `:164-167`), unlike Qwen2. Consequences for us:

- The cache stores `RoPE( k_norm( k_proj ) )`. Stripping RoPE recovers the **normalized**
  key, not the raw projection. This is fine — the mapper is fit on *stored* caches from both
  models, so "content space" here means *normalized-K content space*. Both source and target
  live in the same convention; the linear map absorbs the (fixed, per-model) norm scaling.
- **Do not** try to invert `k_norm` — its weights differ per model and inversion is neither
  needed nor well-posed. We map stored-cache → stored-cache, matching the paper.

### 3.2 RoPE strip/re-apply in vLLM
There is **no built-in inverse** in `vllm/model_executor/layers/rotary_embedding/`. RoPE is
NEOX-style: `o1 = x1·cos − x2·sin ; o2 = x2·cos + x1·sin` (`common.py:134`). The inverse is
the transpose (`sin → −sin`):

```
x1 = o1·cos + o2·sin ;   x2 = o2·cos − o1·sin
```

We build cos/sin from `positions` via the source's `(base, rotary_dim)` and re-apply with the
target's. Since 14B and 32B share `theta=1e6` and `d_h=128`, the source and target rotations
are *identical* here — but we keep the strip/re-apply path general (and it is required to
generalize across context length, which is the real point of §3.3).

---

## 4. vLLM integration architecture

Two phases, mapped onto concrete vLLM extension points found in the codebase.

```
┌─ Phase A: OFFLINE CALIBRATION (one-time, per directed pair) ───────────────┐
│  FineWeb-Edu calib set                                                     │
│      │                                                                     │
│      ├─ run 14B prefill → capture stored K,V per layer  (KVCaptureConnector)│
│      ├─ run 32B prefill → capture stored K,V per layer                     │
│      │                                                                     │
│      ├─ strip RoPE on K (both)                                             │
│      ├─ top-k source selection per target layer  (R² ranking)             │
│      ├─ per-head ridge solve  (K and V)                                    │
│      └─ serialize mapper → cross_model_kv/qwen3_14b_to_32b.safetensors     │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Phase B: SERVING-TIME TRANSFER (per request) ─────────────────────────────┐
│  14B process (producer)                     32B process (consumer)         │
│    prefill → stored KV        ──P2P──▶       CrossModelKVConnector          │
│                                               ├─ get_num_new_matched_tokens │
│                                               │    → claim whole prompt     │
│                                               ├─ strip src RoPE             │
│                                               ├─ apply mapper (batched mm)  │
│                                               ├─ re-apply tgt RoPE          │
│                                               └─ write into paged KV via    │
│                                                    slot_mapping (per layer) │
│                                               32B decodes, prefill SKIPPED  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 The KV Connector is the injection point
vLLM's `KVConnectorBase_V1`
(`vllm/distributed/kv_transfer/kv_connector/v1/base.py:185`) is exactly the seam we need. The
relevant hooks:

- **Scheduler side** — `get_num_new_matched_tokens(request, num_computed_tokens)` (`:488`):
  return the full prompt length so the scheduler treats prefill as "already available
  externally" and **skips it**. `update_state_after_alloc` (`:523`) reserves blocks;
  `build_connector_meta` (`:549`) passes per-request slot mappings to the worker.
- **Worker side** — `start_load_kv(forward_context)` (`:312`) and per-layer
  `wait_for_layer_load(layer_name)` (`:330`): this is where we materialize the mapped target
  KV and write it into the paged buffer using `slot_mapping`. The per-layer hook fires from
  inside attention via `maybe_transfer_kv_layer`
  (`vllm/model_executor/layers/attention/kv_transfer_utils.py:15`).
- **Capture (Phase A)** — `save_kv_layer(layer_name, kv_layer, attn_metadata)` (`:344`) plus
  the `extract_from_kv_cache(kv_cache, slot_mapping, num_tokens)` helper pattern from
  `example_hidden_states_connector.py:36` to pull per-token K/V out of the paged cache.

Model our connectors on **`ExampleHiddenStatesConnector`**
(`.../v1/example_hidden_states_connector.py`, `class ...(KVConnectorBase_V1, SupportsHMA)`):
it already demonstrates slot-mapping-keyed capture, async safetensors writes, and returning a
file path to the client through `kv_transfer_params`. `ExampleConnector`
(`.../v1/example_connector.py`) is the simpler save/load-to-disk template.

Register new connectors in
`vllm/distributed/kv_transfer/kv_connector/factory.py` and select with
`--kv-transfer-config '{"kv_connector":"CrossModelKV","kv_role":"kv_consumer", ...}'`
(`vllm/config/kv_transfer.py:57`).

### 4.2 Paged KV cache layout (for extract + inject) — ROCm backend
> **On ROCm the CUDA `flash_attn.py` backend is not used.** Because we run a KV connector,
> `ROCM_ATTN` is force-disabled (`vllm/platforms/rocm.py:472-475`, gated on
> `use_kv_connector`), so the selected backend is **`ROCM_AITER_UNIFIED_ATTN`** (available on
> gfx950 via AITER) with **`TRITON_ATTN`** as the portable fallback. We pin it with
> `VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN` (or `TRITON_ATTN`).

Both connector-compatible ROCm backends use the same logical layout as CUDA flash-attn — a
per-layer tensor `(B, H, N, 2·D)` split into `key_cache, value_cache` each `(B, N, H, D)`:
- `ROCM_AITER_UNIFIED_ATTN`: `_split_kv_cache` = `kv_cache.transpose(1, 2).split(head_size, -1)`
  (`vllm/v1/attention/backends/rocm_aiter_unified_attn.py:193`); write via
  `ops.reshape_and_cache_flash(...)` (`:338`); KV layout `LBHNC`/`LHBNC` (`:122-129`).
- `TRITON_ATTN`: same split (`triton_attn.py:786`); write via `triton_reshape_and_cache_flash`
  (`:790`) — a **stride-aware** writer that tolerates HND *or* NHD physical strides
  (`:395-421`), so do not assume C-contiguous HND.

Both are `slot_mapping`-indexed (`slot = block_id·block_size + offset`), so **capture** (gather
by slot) and **inject** (scatter by slot) work identically to the CUDA path. `head_dim=128`,
`n_kv=8`. **Avoid `ROCM_ATTN`** for this feature: it uses a legacy KV-outer layout
(`rocm.py:584-590`) that the connector transfer path has not validated.

### 4.3 Cross-GPU / cross-process transport — ROCm
The 14B producer and 32B consumer run as separate processes/GPUs. On ROCm, `torch.distributed`
uses backend `"nccl"`, which maps to **RCCL**, and inter-GPU DMA runs over **xGMI / Infinity
Fabric** (not NVLink). `multi_gpu_comm.py` in this directory works **unchanged** — its
`dist.init_process_group("nccl")` + `send`/`recv` handshake runs on RCCL transparently
(`rocm.py:494` sets `dist_backend="nccl"`). For the MVP we transfer the **source cache** (or
capture-serialized safetensors) and apply the mapper on the consumer.

Transport caveats specific to ROCm:
- **NIXL** has only partial ROCm support; do **not** assume it for the MVP. The AMD-oriented
  **MoRIIO** connector (`.../v1/moriio/moriio_connector.py`) is the closest RDMA analog.
- The lowest-risk path is a **shared-filesystem handoff** (`ExampleConnector` style) or the
  RCCL `send`/`recv` in `multi_gpu_comm.py`; stand up RDMA only if it becomes the bottleneck.
- gfx950 has ample HBM (MI350-class), so 14B + 32B can be **co-resident** on separate GPUs of
  the same node — the offline capture phase can even run them sequentially on one GPU.

### 4.4 Where the mapper math lives
A backend-neutral module `cross_model_kv/mapper.py` (new), lifted from `algorithm.py`:
- `RidgeKVMapper.fit(...)` / `.transfer(...)` — pure linear algebra.
- `strip_rope` / `apply_rope` built from vLLM's `RotaryEmbedding` cos/sin cache
  (`rotary_embedding/base.py:_compute_cos_sin_cache`) so we match vLLM's exact rotation
  convention rather than re-deriving it.
- Serialization: per (target layer, head, K|V) `W` and `b` in a single safetensors file plus a
  JSON sidecar (`selected` source layers per target layer, `k`, `λ`, dtype, model ids).

### 4.5 ROCm specifics (must-read)
Consolidated so nothing is missed when moving off the paper's CUDA/H100 setup:

| Concern | CUDA (paper) | ROCm (this env) | Action |
|---|---|---|---|
| Attention backend | FlashAttention (`flash_attn.py`) | `ROCM_ATTN` **disabled** under KV connector | Pin `VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN`, fallback `TRITON_ATTN` |
| KV write op | `reshape_and_cache_flash` | `ops.reshape_and_cache_flash` / `triton_reshape_and_cache_flash` | Use the selected backend's writer; same slot-mapping semantics |
| KV physical strides | HND | HND **or** NHD (triton) | Use stride-aware gather/scatter; don't assume contiguity |
| Collectives | NCCL / NVLink | RCCL / xGMI (backend still `"nccl"`) | `multi_gpu_comm.py` unchanged |
| RDMA connector | NIXL | NIXL partial; MoRIIO is AMD path | MVP: shared-FS or RCCL P2P |
| FP8 KV dtype | e4m3fn | gfx950 supports OCP e4m3 (unlike MI300 fnuz) | **Keep KV bf16** to sidestep entirely |
| RoPE kernel | CUDA `ops.rotary_embedding` | HIP kernel exists | We strip/apply RoPE in **pure torch** (device-agnostic) — no kernel dependency |
| Reference HW | 8×H100 | 8×gfx950 (MI350-class) | Latency numbers are **not** comparable; report ROCm numbers, keep the *ratio* (speedup) as the headline |

Additional notes:
- **Build/install:** `VLLM_USE_PRECOMPILED=1` in AGENTS.md targets CUDA wheels; on ROCm use the
  ROCm editable build (the running env already has a working `torch 2.10+hip` — reuse it, do not
  reinstall torch). C++/HIP kernel changes are *not* required for this feature (all mapper math
  is torch/Python), which avoids the ROCm incremental-build path entirely.
- **AITER availability** is gated by `is_aiter_found_and_supported()` (`rocm.py:478`); if AITER
  is unavailable at runtime, the design still works on `TRITON_ATTN` with no code change.
- **Determinism:** our strip-RoPE uses vLLM's exact cos/sin (`RotaryEmbedding._compute_cos_sin_cache`),
  computed on-device; validate the round-trip on ROCm in M0 (numerics can differ subtly from CUDA).

---

## 5. Implementation plan (milestones)

### M0 — Scaffolding & mapper core (no vLLM runtime needed)
- Create package `cross_model_kv/` with `mapper.py` (port `algorithm.py`, drop the synthetic
  data path), `rope.py` (strip/apply against vLLM cos/sin), `io.py` (safetensors + JSON).
- Unit tests on synthetic structured caches (reuse `algorithm.py`'s generator) asserting
  recovered `R²` and attention-output cosine, plus RoPE round-trip exactness
  (`strip(apply(x)) == x`).

### M1 — Offline KV capture from real Qwen3
- **First runtime step — verify backend:** launch both models with
  `VLLM_ATTENTION_BACKEND=ROCM_AITER_UNIFIED_ATTN` and assert the selected backend is *not*
  `ROCM_ATTN`; confirm captured K/V shapes `[N, 8, 128]` per layer on gfx950 before proceeding.
- Implement `KVCaptureConnector` (subclass of `KVConnectorBase_V1`) modeled on
  `ExampleHiddenStatesConnector`: on `save_kv_layer`, gather stored K,V for the prompt tokens
  via `slot_mapping` and append to a per-layer shard on disk.
- Driver `scripts/capture_kv.py`: run `vllm.LLM(Qwen3-14B)` and `vllm.LLM(Qwen3-32B)` over the
  same 500 FineWeb-Edu × 1024-token calibration sequences (prompt-only, `max_tokens=1`) with
  the capture connector enabled; write `calib/{14b,32b}/layer_{l}.safetensors`.
- Validate shapes: `[N_tokens, n_kv=8, d_h=128]` per layer; 40 source / 64 target layers.

### M2 — Fit the mapper
- `scripts/fit_mapper.py`: load captured caches, strip RoPE on K, run top-k selection (k=8),
  per-head ridge solve in float32, serialize to
  `cross_model_kv/artifacts/qwen3_14b_to_32b.safetensors`.
- Report in-sample `R²` (expect ~**0.79 K**, **0.65 V** at k=8 per Table 7) and per-target
  selected source layers as a sanity gate before touching serving.

### M3 — Offline injection eval harness (accuracy, no serving skip yet)
This reproduces the paper's retention numbers **without** the full connector, de-risking the
math before the serving integration.
- Implement `CrossModelKVConnector` (consumer) that, given a request, loads the source (14B)
  cache for the prompt, applies the mapper, and writes mapped KV into the 32B paged buffer;
  `get_num_new_matched_tokens` returns the full prompt length to skip prefill.
- Prefix-conditioned protocol (paper Appendix D): score continuation tokens conditioned on a
  mapped-prefix cache vs. the target's own-prefill cache.
- Wire into the existing eval harnesses:
  - `lm_eval` with `model="vllm"` (offline), mirroring
    `tests/entrypoints/llm/test_accuracy.py:42` and the KV-connector precedent
    `tests/v1/kv_connector/nixl_integration/test_accuracy.py:78`.
  - `tests/evals/gsm8k` config-driven runner for GSM8K.

### M4 — Serving-time transfer + latency
- Two-process disaggregated run: 14B producer + 32B consumer, transport per `multi_gpu_comm.py`
  (or shared-FS MVP). Confirm the 32B **skips prefill** end-to-end.
- Latency harness: mapper-apply vs. 32B re-prefill across seq len 64 → 32,768
  (eager, bf16), reproducing Table 5 shape.

### M5 — Report & polish
- Retention table + latency table vs. paper; ablations (k sweep 1→8, RoPE-stripped vs coupled).
- Docs, config plumbing, and tests green under `pre-commit`.

---

## 6. Evaluation plan (Qwen3 14B → 32B)

**Benchmarks** (lm-eval defaults; paper Table 1/13):

| Benchmark | Metric | Shots | Paper transfer / standalone (retention) |
|---|---|---|---|
| ARC-Challenge | acc_norm | 0 | 61.60 / 60.49* → **101.0%** |
| HellaSwag | acc_norm | 0 | 80.70 / 82.65 → **97.6%** |
| WinoGrande | acc | 0 | 68.98 / 70.01 → **98.5%** |
| MMLU | acc | 5 | 78.09 / 82.17 → **95.0%** |
| GSM8K | acc | 8 (CoT) | 90.98 / 95.15 → **95.6%** |
| WikiText-2 | ppl | prefix-cond. | 7.33 (vs 6.79 standalone) |

\* Retention is vs. the **32B target** standalone; **Avg 97.6%**. (Paper reports retention
against target standalone accuracy; see Table 13.)

**Metric:** `retention = transfer_acc / target_standalone_acc × 100%`, plus floor-normalized
`(acc − chance)/(target − chance)`. Also report **attention-output cosine** (the paper's best
cross-pair predictor; `algorithm.py:attention_cosine`) as an internal diagnostic.

**Latency:** mapper-apply vs. 32B re-prefill, 50 warmup / 30 timed, bf16, eager, on
`ROCM_AITER_UNIFIED_ATTN`. Include cross-GPU (xGMI/RCCL) transfer time in the "transfer"
column. **Absolute ms will differ from the paper's H100 table** — report the ROCm numbers and
use the *speedup ratio* (paper: 4–25× over 64–32k tokens) as the comparable headline.

**Success criteria for the implementation:**
1. In-sample fit `R²` within ~0.02 of paper (K≈0.79, V≈0.65 at k=8).
2. Avg accuracy retention ≥ ~95% across the five benchmarks (paper: 97.6%).
3. Demonstrated prefill skip on the 32B receiver with measured speedup > 3× at ≥8k tokens.

---

## 7. Risks & open questions

- **q_norm/k_norm convention.** Handled by mapping stored-cache→stored-cache (§3.1); flagged so
  reviewers don't attempt to invert the norm. Verify empirically that content-space fit `R²`
  matches the paper — if it's materially lower, revisit whether K should be de-normed.
- **Connector prefill-skip semantics.** `get_num_new_matched_tokens` returning the full prompt
  must interact correctly with block allocation and the HMA path (`SupportsHMA`). Follow
  `ExampleHiddenStatesConnector` precisely; add an assertion that 0 prefill tokens are computed.
- **Cache-layout coupling (ROCm).** We index via `slot_mapping` and the ROCm connector-
  compatible K/V split (`rocm_aiter_unified_attn.py:193` / `triton_attn.py:786`). Scope the MVP
  to `ROCM_AITER_UNIFIED_ATTN` (+`TRITON_ATTN` fallback) and **hard-fail if `ROCM_ATTN` is
  selected** (its legacy layout is unvalidated for connectors, `rocm.py:472-475,584-590`).
  Triton's writer allows NHD *or* HND strides — use stride-aware indexing, not `.reshape`.
- **ROCm numerics.** Validate strip/apply-RoPE round-trip and mapper `R²` on gfx950 in M0;
  bf16 accumulation and AITER kernels can differ subtly from the paper's CUDA/H100 fit.
- **Calibration data.** Paper uses FineWeb-Edu; domain shift costs up to ~5 pp (Table 8). Use
  FineWeb-Edu to match, keep the corpus configurable.
- **Two-model VRAM.** 14B + 32B co-resident needs a multi-GPU node; the offline capture phase
  can run them sequentially to fit smaller setups.
- **Scope.** Dense full-attention, within-family, matched-KV only — matching the paper's stated
  limitations. No mismatched-KV, no hybrid/SSM.

---

## 8. Deliverables checklist

- [x] `cross_model_kv/{mapper,rope,io}.py` + unit tests (M0)
- [x] `KVCaptureConnector` + `scripts/capture_kv.py` (M1)
- [x] `scripts/fit_mapper.py` + serialized artifact (M2)
- [x] `CrossModelKVConnector` (consumer, prefill-skip) + shared `capture_store` transport (M3)
- [ ] lm_eval/GSM8K wiring + two-process serving demo + latency harness (M4)
- [ ] Results report: retention + latency vs. paper, k-ablation (M5)

---

## 9. File/line reference index (grounding)

- Qwen3 attention, q_norm/k_norm, RoPE apply: `vllm/model_executor/models/qwen3.py:65,150-151,161-167`
- Attention layer / KV write: `vllm/model_executor/layers/attention/attention.py:225,485,546`
- **ROCm backend selection / KV-connector gating:** `vllm/platforms/rocm.py:471-485,494,584-590`
- **ROCm paged K/V split & write (AITER):** `vllm/v1/attention/backends/rocm_aiter_unified_attn.py:122-129,193,338`
- **ROCm paged K/V split & write (Triton fallback):** `vllm/v1/attention/backends/triton_attn.py:786,790,395-421`
- CUDA reference (not used on ROCm): `vllm/v1/attention/backends/flash_attn.py:1454,1463`
- KV cache spec / layout: `vllm/v1/kv_cache_interface.py:478,538`; `vllm/v1/kv_cache_layout.py:12`
- RoPE base / cos-sin: `vllm/model_executor/layers/rotary_embedding/base.py:80-103`; `common.py:134`
- Connector base interface: `vllm/distributed/kv_transfer/kv_connector/v1/base.py:185,312,330,344,488,523,549`
- Per-layer transfer hook: `vllm/model_executor/layers/attention/kv_transfer_utils.py:15`
- Capture/inject reference: `.../v1/example_hidden_states_connector.py:36`; `.../v1/example_connector.py`
- Connector registry / config: `.../kv_connector/factory.py:27`; `vllm/config/kv_transfer.py:57`
- Scheduler hook sites: `vllm/v1/core/sched/scheduler.py:938,1188,1462`
- Eval precedents: `tests/entrypoints/llm/test_accuracy.py:42`; `tests/v1/kv_connector/nixl_integration/test_accuracy.py:78`; `tests/evals/gsm8k/`
