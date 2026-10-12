"""Cross-model KV cache transfer — closed-form ridge mapper for vLLM.

Implements Heo et al., "Cross-Model KV Cache Transfer in LLM Families:
A Closed-Form Linear Mapping for Prefill Reuse" (arXiv:2608.03893), for the
Qwen3 14B -> 32B matched-KV pair on ROCm.

Public surface:
    MapperConfig    -- mapper hyper-parameters and pair geometry
    RidgeKVMapper   -- top-k source selection + per-head closed-form ridge
    RopeCodec       -- position-free (RoPE-stripped) content-space transform
    save_mapper / load_mapper -- safetensors + JSON (de)serialization
    attention_cosine          -- the paper's cross-pair retention predictor
"""

from cross_model_kv.config import MapperConfig
from cross_model_kv.diagnostics import attention_cosine, attention_output
from cross_model_kv.io import load_mapper, save_mapper
from cross_model_kv.mapper import RidgeKVMapper
from cross_model_kv.ridge import r2_score, ridge_fit
from cross_model_kv.rope import RopeCodec

__all__ = [
    "MapperConfig",
    "RidgeKVMapper",
    "RopeCodec",
    "attention_cosine",
    "attention_output",
    "load_mapper",
    "save_mapper",
    "r2_score",
    "ridge_fit",
]
