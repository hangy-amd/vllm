"""Mapper configuration: pair geometry and closed-form hyper-parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MapperConfig:
    """Geometry of a directed transfer pair plus mapper hyper-parameters.

    "Matched-KV" (the paper's setting, and the Qwen3 14B -> 32B pair) means
    source and target agree on KV-head count and per-head dimension, even though
    depth and parameter count differ. The mapper imposes no constraint on the
    source/target layer counts; cross-layer selection bridges the depth gap.
    """

    # ---- pair geometry -------------------------------------------------------
    num_source_layers: int
    num_target_layers: int
    num_kv_heads: int
    head_dim: int

    # RoPE base ("theta") for each model. Qwen3 14B and 32B both use 1e6, but we
    # keep them separate so content-space mapping generalises to pairs that
    # differ here.
    source_rope_theta: float = 1_000_000.0
    target_rope_theta: float = 1_000_000.0
    # Rotary dimension. For Qwen3 the full head is rotated (partial factor 1.0).
    rotary_dim: int | None = None

    # ---- mapper hyper-parameters (paper defaults) ----------------------------
    top_k: int = 8              # source layers concatenated per target layer
    ridge_lambda: float = 1e-2  # Tikhonov term

    # ---- provenance (free-form, serialised into the JSON sidecar) ------------
    source_model: str = ""
    target_model: str = ""

    # ---- misc ----------------------------------------------------------------
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rotary_dim is None:
            self.rotary_dim = self.head_dim
        if self.rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {self.rotary_dim}")
        if self.rotary_dim > self.head_dim:
            raise ValueError(
                f"rotary_dim ({self.rotary_dim}) exceeds head_dim ({self.head_dim})"
            )
        if not 1 <= self.top_k <= self.num_source_layers:
            raise ValueError(
                f"top_k ({self.top_k}) must be in [1, num_source_layers="
                f"{self.num_source_layers}]"
            )

    @property
    def source_layer_width(self) -> int:
        """Flattened width of one source layer's KV (all heads concatenated)."""
        return self.num_kv_heads * self.head_dim

    @property
    def feature_dim(self) -> int:
        """Per-head ridge input width: top_k source layers concatenated."""
        return self.top_k * self.source_layer_width

    def to_dict(self) -> dict:
        return {
            "num_source_layers": self.num_source_layers,
            "num_target_layers": self.num_target_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "source_rope_theta": self.source_rope_theta,
            "target_rope_theta": self.target_rope_theta,
            "rotary_dim": self.rotary_dim,
            "top_k": self.top_k,
            "ridge_lambda": self.ridge_lambda,
            "source_model": self.source_model,
            "target_model": self.target_model,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict) -> MapperConfig:
        known = {f: data[f] for f in cls.__dataclass_fields__ if f in data}
        return cls(**known)
