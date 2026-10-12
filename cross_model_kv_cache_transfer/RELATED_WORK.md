# Related Work — Cross-Model KV Cache Transfer & Linear-Attention Analogues

**Scope:** Literature survey supporting the closed-form per-head ridge KV mapper (Qwen3-14B → Qwen3-32B) documented in [`REPORT.md`](REPORT.md).
**Anchor paper:** Heo, Shafipour, Zhao, Golub, Kamani, Borkar, Chandran, Zardoshti, Darvish Rouhani — *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse* ([arXiv:2608.03893](https://arxiv.org/abs/2608.03893)).
**Compiled:** 2026-09-15

---

## 0. How to read this document

The field splits along two axes that are easy to conflate:

- **Axis 1 — quantity moved:** softmax KV cache vs. attention patterns vs. linear/SSM recurrent state.
- **Axis 2 — mechanism:** same-model reuse vs. learned cross-model transfer vs. closed-form cross-model mapping.

The anchor paper occupies **closed-form / cross-model / KV-values**. §1–§2 place its direct competitors; §3–§5 cover the surrounding softmax-attention literature; §6–§7 cover the linear-attention / hybrid world, which targets a *different* problem with *different* tools. §8 is the coverage matrix and the open gap.

**One-line finding:** every method the anchor paper compares against targets full (softmax) attention by construction — "KV cache transfer" only has meaning where a growing KV cache exists. Linear-attention models replace that cache with a fixed-size recurrent state, so the paradigm does not port mechanically; the analogous work there is offline distillation (not runtime mapping) and same-model snapshot reuse (not cross-scale mapping). The closed-form cross-model *state* transfer cell is empty.

---

## 1. Direct competitors — cross-model KV cache transfer (anchor paper, Table 6)

The anchor paper's contribution is defined relative to four methods. Its claim: it is the **only** one that is simultaneously *gradient-free*, *cross-scale*, *transfers KV values* (not attention patterns), and *closed-form*.

### 1.1 C2C — Cache-to-Cache
*Direct Semantic Communication Between Large Language Models* — Fu, Min, Zhang, Yan, Dai, Ouyang, Wang. ICLR 2026. [arXiv:2510.03215](https://arxiv.org/abs/2510.03215)

- **Mechanism:** a neural-network projector fuses the source model's KV-cache into the target's, with a learnable gating mechanism selecting which target layers receive cache.
- **Results:** 6.4–14.2% higher average accuracy than individual models; ~3.1–5.4% over text-based communication; ~2.5× latency speedup.
- **vs. anchor:** transfers KV values and is cross-scale, but the projector is **learned (gradient-based), not closed-form**. Closest competitor; the sharpest head-to-head benchmark target.

### 1.2 LatentAlign — Latent Space Communication via K-V Cache Alignment
Dery, Yahav, Prior, Feng, Shen, Szlam. 2026. [arXiv:2601.06123](https://arxiv.org/abs/2601.06123)

- **Mechanism:** adapters translate each model's KV into/out of a shared latent space — "a high-bandwidth channel for collaboration without altering the underlying pre-trained parameters." Also transfers soft prompts across models.
- **Evidence:** Gemma-2 models; improves both inter-model communication and individual performance.
- **vs. anchor:** KV-value transfer, cross-scale, but adapters are **trained**. Not closed-form.

### 1.3 IAM — Inference through Attention Mapping between Different-scale LLMs
Zhao, Li, Zhao. ACL 2025. [ACL Anthology 2025.acl-long.959](https://aclanthology.org/2025.acl-long.959/) · [arXiv:2507.11953](https://arxiv.org/abs/2507.11953)

- **Mechanism:** cosine-similarity mapping between small-model (SLM) and large-model (LLM) attention heads established during prefill; during decode, the LLM's mapped layers reuse the SLM's **attention matrices**. Delayed-establishment mechanism + short Alpaca instruction-tuning to adapt to "borrowed" attention.
- **Results:** ~15% faster prefill, 22.1% less KV usage on Qwen2 / Llama-3, evaluated on MMLU / HotpotQA / GovReport; orthogonal to token-eviction methods (H₂O).
- **Caveats (independent commentary):** the mapping step runs *both* models fully during prefill (more work, not less, in that phase); incompatible with FlashAttention.
- **vs. anchor:** gradient-free-ish and cross-scale, but transfers **attention patterns, not KV values** — a different quantity, and it does not skip target prefill.

### 1.4 DroidSpeak — KV Cache Sharing Across Fine-tuned Model Variants
Liu, Huang, Yao, Feng, Gu, Du, Li, Cheng, Jiang, Lu, Musuvathi, Choukse. USENIX NSDI 2026. [USENIX](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) · [arXiv:2411.02820](https://arxiv.org/abs/2411.02820) (arXiv title: *KV Cache Sharing for Cross-LLM Communication and Multi-LLM Serving*)

- **Mechanism:** for two models fine-tuned from the same base, selectively **recompute the ~10% "critical" layers** and reuse the rest of the KV cache; pipeline the recompute with reused-KV loading.
- **Results:** up to 4× throughput, ~3.1× faster prefill (TTFT), negligible F1 / Rouge-L / code-similarity loss.
- **vs. anchor:** transfers KV values and is gradient-free, but works **only between architecturally identical models** (same hidden size, layer count, head config) — not cross-scale, no mapping. Its 3.1× TTFT is the systems bar the anchor paper's MVP (1.82–2.34×) currently sits below (`REPORT.md` §5.3), and its pipelined layer-load is the concrete engineering target for closing that gap.

---

## 2. Theoretical foundation — representation transferability

Why a low-rank linear map between two sibling models can hit R²≈0.95 on K.

- **The Platonic Representation Hypothesis** — Huh, Cheung, Wang, Isola. ICML 2024. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987). Neural-net representations converge toward "a shared statistical model of reality" as models scale; vision and language models measure inter-datapoint distance increasingly alike. **The conceptual keystone** — the license for expecting a clean cross-model map to exist.
- **Linear Representation Transferability Hypothesis** — Bello, Das, Zeng, Yin, Liu. 2025. [arXiv:2506.00653](https://arxiv.org/abs/2506.00653). Leverage small models to steer large models via linear maps.
- **Transferring Linear Features Across Language Models with Model Stitching** — Chen, Merullo, Stolfo, Pavlick. NeurIPS 2025.
- **Cross-Model Transferability among LLMs on the Platonic Representations of Concepts** — Huang, Huang, Feng, Lei, Lv. ACL 2025.

## 3. KV reuse & prefill acceleration in serving (softmax attention)

These all avoid full prefill cost, from different angles. Relative to the ridge mapper: this work reuses KV **within one model across requests** (Prompt Cache) or **shrinks the prompt** before prefill (Speculative Prefill family); only ICaRus and PrefillShare are genuinely multi-model, and both achieve it by *constraining the models to emit identical KV* rather than *mapping* between independently trained models.

**One-line takeaway:** the multi-model methods here make sharing work by *engineering the models to agree*; the compression methods reduce prefill *work*; only the anchor (and C2C) *reconstruct* one model's KV from another's without co-design.

### 3.1 Prompt Cache — Modular Attention Reuse for Low-Latency Inference
Gim et al. MLSys 2024. The foundational "reuse KV across requests" work. Precomputes the KV of **reusable text segments** (system prompt, few-shot block, a quoted document) once and splices them into later prompts so overlapping regions skip prefill. Contributes a **Prompt Markup Language (PML)** declaring reusable modules, and handles the **position-dependence problem** — cached KV is only reusable if its positional encoding lines up at reuse time — by assigning modules fixed position ranges.

- **vs. anchor:** the *intra-model, exact-text* version of what the anchor does *cross-model, learned*. Its positional-alignment problem is the same class the anchor solves via RoPE strip/reapply. Cannot help when the two requests use different models.

### 3.2 SwiftKV — Fast Prefill-Optimized Inference with Knowledge-Preserving Model Transformation
Qiao, Yao, Rajbhandari, He. EMNLP 2025 (Snowflake). Cuts prefill compute with a model transformation: **SingleInputKV** rewires the network so upper-layer KV is computed from an *earlier* layer's hidden state, skipping much of upper-layer prefill; paired with light distillation to recover quality. ~50% prefill FLOP reduction with minimal accuracy loss on Llama.

- **vs. anchor:** a *single-model architectural* prefill-skip that modifies weights (needs the distill step); the anchor gets prefill-skip by *borrowing another model's KV*, closed form, leaving both models untouched.

### 3.3 Speculative Prefill — Turbocharging TTFT with Lightweight Token Importance Estimation
Liu, Chen, Zhang. ICML 2025. The "shrink the prompt" family: a **lightweight model scores token importance** via attention, drops low-importance tokens, prefills only the survivors. Training-free, but assumes a **same-tokenizer** draft model.

- **vs. anchor:** orthogonal and composable — reduces *how many tokens* are prefilled; the anchor reduces *which model has to prefill them at all*. Compress first, then map.

### 3.4 Cross-Family Speculative Prefill — Training-Free Long-Context Compression with Small Draft Models
Upasani et al. 2026. [arXiv:2603.02631](https://arxiv.org/abs/2603.02631). Generalizes §3.3 to **draft and target from different families** (Qwen / Llama / DeepSeek), no shared tokenizer. Sidesteps cross-tokenizer positional alignment by assigning **fresh contiguous position IDs** to the compressed prompt (cf. the anchor's RoPE reapply). Retains 90–100% of full-prompt accuracy; on RULER, 128k→16k compression cut TTFT ~18× (46s→2.5s); lets a 32k-context target consume a 128k input via a long-context draft.

- **vs. anchor — strong independent evidence:** its headline finding — **attention-based token importance transfers reliably across families despite architecture/tokenizer differences** — supports the anchor's cross-model-signal-transfer premise. But it transfers a *token-importance mask*, a far lower-bandwidth object than the KV values the anchor reconstructs. The "cheap cousin" of the anchor's method.

### 3.5 ICaRus — Identical Cache Reuse for Efficient Multi-Model Inference
Woo et al. ICLR 2026 (NAVER Cloud). [arXiv:2603.13281](https://arxiv.org/abs/2603.13281). Closest in *goal* to the anchor, opposite in *method*. Decomposes a decoder-only transformer into a **logical encoder** (produces KV) and **logical decoder** (predicts tokens), **freezes the encoder** across all task specialists, and fine-tunes only the decoder with **LoRA adapters** — so every specialist emits *bit-identical KV* for a prompt and KV is shared verbatim across agents. Overlaps next-step KV generation with current-token decoding to hide the split. Up to **11.1× lower P95 latency, 3.8× higher throughput** in an 8-agent ReAct/Reflexion workflow on HotpotQA, integrated into vLLM.

- **vs. anchor:** ICaRus makes sharing trivial by *forcing* models to agree (shared frozen encoder), requiring you to own and co-train the family; the anchor maps between independently trained, different-scale off-the-shelf models (Qwen3-14B → 32B).

### 3.6 PrefillShare — A Shared Prefill Module for KV Reuse in Multi-LLM Disaggregated Serving
Woo et al. 2026. [arXiv:2602.12029](https://arxiv.org/abs/2602.12029). Same NAVER group as ICaRus. A **single shared prefill stage** whose KV feeds multiple downstream models in a disaggregated (prefill/decode-split) deployment, rather than a learned map between independent models. *(Lower-confidence summary — description is directional.)*

- **vs. anchor:** same "share the prefill" spirit as ICaRus, architected for disaggregated serving — a systems complement to, not a competitor of, the mapping approach.

## 4. KV cache compression / cross-layer structure

This cluster establishes the **empirical regularity the anchor depends on**: KV across layers is highly redundant and low-rank. The mapper's top-k=8 cross-layer source selection works precisely because a target layer's KV is well-predicted by a small set of source layers. These papers are independent proof that such structure exists.

**One-line takeaway:** CLA, xKV, and Layer-Condensed independently show KV is **low-rank and highly redundant across layers** — the exact assumption that makes the anchor's closed-form per-head, top-k-cross-layer ridge fit both possible and cheap.

### 4.1 Cross-Layer Attention (CLA) — Reducing Transformer KV Cache Size
Brandon, Mishra, Nrusimha, Panda, Ragan-Kelley. NeurIPS 2024 (MIT). Extends MQA/GQA *across depth*: **groups of adjacent layers share one KV projection**, cutting KV ~2× on top of MQA with near-neutral quality.

- **vs. anchor:** direct architectural evidence that **adjacent layers' KV is redundant enough to share** — the structural fact the anchor's cross-layer regression exploits at inference time instead of baking into the architecture (pretrained shared cache vs. post-hoc ridge fit).

### 4.2 xKV — Cross-Layer SVD for KV-Cache Compression
Chang et al. 2025 (Cornell / Abdelfattah lab). [arXiv:2503.18893](https://arxiv.org/abs/2503.18893) · [code](https://github.com/abdelfattah-lab/xKV). The sharpest characterization of the cross-layer structure, and the most methodologically relevant. Via **Centered Kernel Alignment**: adjacent layers' KV have *low cosine similarity* token-by-token **but highly aligned dominant singular vectors**. So per-token cross-layer copy fails, but a **shared low-rank subspace across layers** captures most of the signal — extracted by a cross-layer SVD, plug-and-play, no fine-tuning. Up to **8× KV compression at ~88.5% RULER** on Llama-3.1-8B (vs. 45.7% for single-layer SVD at the same ratio), 4.23× end-to-end speedup with Selective Reconstruction, MLA-compatible (DeepSeek), stackable with 4-bit quant to 25.6× total.

- **vs. anchor — the one to engage directly:** xKV is the *compression-flavored twin* of the anchor. Both bet a **low-rank cross-layer linear structure** captures the KV; the anchor uses it to *map source→target across models*, xKV to *compress within one model*. Its CKA result ("cosine sim low, dominant subspace shared") is a caution worth citing — it implies the per-head ridge is capturing subspace alignment, and that **singular-vector alignment may select top-k sources better than raw similarity**.

### 4.3 Layer-Condensed KV Cache
Wu, Tu. ACL 2024. The most aggressive cross-layer sharing: **compute and cache KV for only a few layers (in the limit, just the top layer) and have all query layers attend to those**, reordering computation so most layers skip KV entirely. Up to ~26× throughput in the extreme config with modest quality loss, tunable by how many layers keep their own KV.

- **vs. anchor:** the strongest statement of "you don't need every layer's KV — most is recoverable/redundant." Where Layer-Condensed *drops* most layers' KV, the anchor *reconstructs* the target's per-layer KV from a handful of source layers — same redundancy premise, opposite operation (discard vs. regress).

## 5. Supporting infrastructure & model families (cited by anchor)

- **SGLang — Efficient Execution of Structured LM Programs** — Zheng et al. NeurIPS 2024 (serving-runtime peer to the vLLM integration).
- **RoFormer / RoPE** — Su, Ahmed, Lu, Pan, Wen, Liu. Neurocomputing 2024 (directly relevant to the source-RoPE-strip → map → target-RoPE-reapply handling).
- Model families: **Qwen3** (Yang et al. 2025, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)), **Llama 3** (Grattafiori et al. 2024), **Nemotron 3** (Blakeman et al. 2025), **GPT-5** (OpenAI 2025).
- Benchmarks cited: ARC, GSM8K, MMLU, CoQA, WinoGrande, HellaSwag; Chain-of-Thought (Wei et al. NeurIPS 2022). Evaluation infrastructure, not field contributions.

---

## 6. Linear-attention analogue — Thread A: cross-*model* transfer = offline distillation

Moving knowledge *between* a transformer and a Mamba/linear-attention model is done by **distillation with a linearized-attention "bridge"** at training time, weight-level — not by runtime KV-style mapping. The recurring finding across this thread is that *naïve* cross-architecture transfer fails and needs a bridge, which is itself evidence that the clean linear regression that works on softmax K/V would not transplant to state space without a learned intermediary.

- **Attention to Mamba — A Recipe for Cross-Architecture Distillation** — 2026. [arXiv:2604.14191](https://arxiv.org/abs/2604.14191) · [OpenReview](https://openreview.net/forum?id=EXtPGwYdWe). Two-stage: transformer → linearized attention (kernel-trick adaptation) → Mamba. Distilled Mamba preserves Pythia-1B downstream performance (perplexity 14.11 vs. teacher 13.86). Explicitly notes naïve Transformer→Mamba distillation fails, usually patched with hybrids.
- **Data-Efficient Any-Transformer-to-Mamba Distillation via Attention Bridge (CAB)** — 2025. [arXiv:2510.19266](https://arxiv.org/html/2510.19266). Token-level supervision through a lightweight bridge + flexible layer-wise alignment; more data-efficient than output-level KD.
- **Precursors (2024):** MOHAWK / *Transformers to SSMs: Distilling Quadratic Knowledge*; *The Mamba in the Llama* (init Mamba from Llama weights + distill); *LoLCATs* (linearize attention via low-rank adaptation).

**vs. anchor:** gradient-based, offline, weight-level — the opposite of a closed-form runtime map.

## 7. Linear-attention analogue — Thread B: runtime state *reuse* = same-model snapshot-copy

Directly touches the local vLLM / MiniMax-M3 work. Frameworks now do prefix caching for SSM/linear layers, but only *within one model*, and via a mechanism unlike KV transfer.

- **Structural fact:** KV cache and SSM state are **mathematically incompatible representations** — nobody *converts* KV → state. SSM state is a fixed-size, in-place-updated summary; effective `block_size = 1` (each block is a whole-sequence snapshot), so it cannot be split at a token boundary or referenced immutably.
- **Mamba** — Gu, Dao. 2023. [arXiv:2312.00752](https://arxiv.org/abs/2312.00752). Selective state; content-based routing that fixed input-independent SSMs' recall weakness. Background for why the state is compressed and lossy.
- **SGLang MambaRadixCache** — hybrid radix tree; on a hit, copies the whole Mamba state snapshot into a fresh buffer (in-place updates would corrupt a shared prefix); request-level Mamba pool vs. token-level KV pool. [PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) · [Alibaba Cloud writeup](https://www.alibabacloud.com/blog/hybrid-model-support-%7C-sglangs-support-scheme-for-hybrid-architecture-models-like-mamba-transformer_602857).
- **vLLM Hybrid Memory Allocator (HMA) + NIXL connector** — groups layers by type; pads FA and SSM pages to a uniform byte size; adds dedicated state-transfer channels for disaggregated hybrid serving. [vLLM blog 2026-04-21](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) · [RFC #17140](https://github.com/vllm-project/vllm/issues/17140).
- **LMCache** — reinterprets the recurrent state as an *opaque page* for model-agnostic reuse; `--separate-object-groups` splits full-attention vs. mamba/sliding-window layers into distinct object groups. [LMCache docs](https://docs.lmcache.ai/mp/hybrid_models.html).
- **DUET — Disaggregated Hybrid Mamba-Transformer LLMs with Prefill- and Decode-Specific Packages** — 2026. [arXiv:2603.15530](https://arxiv.org/html/2603.15530v1).
- **MemMamba — Rethinking Memory Patterns in State Space Models** — 2025. [arXiv:2510.03279](https://arxiv.org/pdf/2510.03279). Early-info decays exponentially in intra/inter-layer propagation; adds lightweight state summarization + cross-layer/cross-token attention to preserve and reuse salient state. (Within-model, not cross-model, but relevant to how lossy the state is.)
- **Cost of missing this (MLX):** without prefix caching, a 40K-token context takes ~200 s vs. ~5 s with reuse. [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980).

**vs. anchor:** identical-model prefix reuse via snapshot-copy — no mapping, no cross-scale, no cross-model. This is the DroidSpeak corner (reuse, same architecture) extended to SSM state, not the C2C / anchor corner (map across different models).

---

## 8. Coverage matrix and the open gap

|                          | Softmax attention (KV)            | Linear / SSM (recurrent state)                          |
|--------------------------|-----------------------------------|---------------------------------------------------------|
| **Same-model reuse**     | Prefix caching (vLLM / SGLang)    | Snapshot-copy — MambaRadixCache, HMA, LMCache ✅          |
| **Cross-model, learned** | C2C, LatentAlign                  | Distillation + linear-attn bridge — Attn→Mamba, CAB ✅    |
| **Cross-model, closed-form** | **Anchor: per-head ridge mapper** ✅ | **empty** ⬅ open frontier                            |

Feature comparison of the direct KV-transfer competitors (anchor paper Table 6):

| Method       | Gradient-free | Cross-scale | Transfers KV values | Closed-form |
|--------------|:---:|:---:|:---:|:---:|
| C2C          | ✗ | ✓ | ✓ | ✗ |
| LatentAlign  | ✗ | ✓ | ✓ | ✗ |
| IAM          | ✓ | ✓ | ✗ (attention) | ✗ |
| DroidSpeak   | ✓ | ✗ | ✓ | ✗ |
| **Anchor / this work** | ✓ | ✓ | ✓ | ✓ |

**The gap.** The bottom-right cell — *closed-form, gradient-free, cross-model recurrent-state transfer* — is unoccupied. The field's own framing explains why: "KV and SSM state are mathematically incompatible," and "naïve cross-architecture transfer fails." A closed-form `S_src → S_tgt` map between two *different* hybrids would contradict both assumptions and is a natural extension paper.

**Implications for the anchor result:**

1. **Accuracy claim is well-supported** — the Platonic hypothesis plus IAM's "attention patterns are similar across scales" finding give independent grounding for the linear fit (R²=0.947 on K).
2. **Latency is the exposed flank** — DroidSpeak already shows 3.1× TTFT with the same *class* of technique on identical models. The §5.3 framing (transport + eager execution, not method) is the right defense; citing DroidSpeak's pipelined layer-load as the target strengthens it.
3. **Lead with the three-way novelty** — *closed-form + cross-scale + KV-values*; no cited method has all three. C2C is the sharpest comparison (shares two of three, differs only by being learned).

**Hybrid / MiniMax-M3 extension (open question).** The anchor's ridge mapper already handles the full-attention layers cross-scale. For a hybrid target, the SSM layers need either (a) snapshot-copy à la SGLang if source/target share SSM geometry (head count, state dim, conv width), or (b) a genuinely new per-channel state map if they do not. Whether a closed-form state fit reaches usable R² on the *compressed* state — which already discards history the KV cache retains — is untested and is the concrete next experiment.

---

## Appendix — consolidated source links

Anchor & competitors
- Anchor — [arXiv:2608.03893](https://arxiv.org/abs/2608.03893)
- C2C — [arXiv:2510.03215](https://arxiv.org/abs/2510.03215)
- LatentAlign — [arXiv:2601.06123](https://arxiv.org/abs/2601.06123)
- IAM — [ACL 2025](https://aclanthology.org/2025.acl-long.959/) · [arXiv:2507.11953](https://arxiv.org/abs/2507.11953)
- DroidSpeak — [USENIX NSDI '26](https://www.usenix.org/conference/nsdi26/presentation/liu-yuhan) · [arXiv:2411.02820](https://arxiv.org/abs/2411.02820)

Foundations / compression
- Platonic Representation Hypothesis — [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)
- Linear Representation Transferability — [arXiv:2506.00653](https://arxiv.org/abs/2506.00653)
- xKV — [arXiv:2503.18893](https://arxiv.org/abs/2503.18893)
- PrefillShare — [arXiv:2602.12029](https://arxiv.org/abs/2602.12029)
- Cross-Family Speculative Prefill — [arXiv:2603.02631](https://arxiv.org/abs/2603.02631)
- ICaRus — [arXiv:2603.13281](https://arxiv.org/abs/2603.13281)
- Qwen3 — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)

Linear / hybrid
- Attention to Mamba — [arXiv:2604.14191](https://arxiv.org/abs/2604.14191) · [OpenReview](https://openreview.net/forum?id=EXtPGwYdWe)
- Attention Bridge (CAB) — [arXiv:2510.19266](https://arxiv.org/html/2510.19266)
- Mamba — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
- MemMamba — [arXiv:2510.03279](https://arxiv.org/pdf/2510.03279)
- DUET — [arXiv:2603.15530](https://arxiv.org/html/2603.15530v1)
- vLLM hybrid SSM disagg — [blog](https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg) · [RFC #17140](https://github.com/vllm-project/vllm/issues/17140)
- SGLang hybrid — [PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- LMCache hybrid — [docs](https://docs.lmcache.ai/mp/hybrid_models.html)
- MLX prefix-cache cost — [mlx-lm #980](https://github.com/ml-explore/mlx-lm/issues/980)
