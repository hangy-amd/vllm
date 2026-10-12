# ROCm: shadow the incompatible CUDA-built flash_attn so vLLM's
# `suppress(ModuleNotFoundError)` fallback engages instead of crashing on the
# ImportError raised by the stray flash_attn_2_cuda .so. Process-local (via
# PYTHONPATH); does not touch the shared system install.
raise ModuleNotFoundError("flash_attn disabled on ROCm (cross_model_kv shim)")
