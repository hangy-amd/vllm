"""Content-space (RoPE-stripped) transform (paper Sec 3.3).

Keys are stored *post-RoPE*. We strip the position-dependent rotation before
fitting so the map lives in a position-free content space, then re-apply the
target's rotation at transfer time. This matches vLLM's NEOX rotary convention
(``vllm/model_executor/layers/rotary_embedding``): a "half-rotation" where the
first ``rotary_dim/2`` channels pair with the second ``rotary_dim/2``. Since
RoPE is orthogonal, the strip is exact.

For Qwen3 the full head is rotated (``rotary_dim == head_dim``); channels beyond
``rotary_dim`` (partial-rotary models) pass through unchanged.

Layout: tensors are ``[..., T, H, D]`` — token axis at ``-3``, head axis at
``-2``, per-head channel axis at ``-1``. cos/sin broadcast over leading and head
axes.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def _rotate_half(x: Tensor) -> Tensor:
    """NEOX half-rotation: ``[x1, x2] -> [-x2, x1]`` along the last axis."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RopeCodec:
    """Builds NEOX cos/sin tables for a model and applies/strips the rotation.

    One codec is fit per model (source or target) from its ``rope_theta`` and
    ``rotary_dim``. Tables are cached and lazily extended as longer position
    ranges are requested.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int | None = None,
        theta: float = 1_000_000.0,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        rotary_dim = rotary_dim or head_dim
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.theta = float(theta)
        self.device = torch.device(device)
        self.dtype = dtype
        half = rotary_dim // 2
        # inv_freq matches vllm RotaryEmbedding._compute_inv_freq.
        self.inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, rotary_dim, 2, device=self.device).float() / rotary_dim)
        )
        assert self.inv_freq.numel() == half

    def cos_sin(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Doubled cos/sin tables ``[T, 1, rotary_dim]`` for ``positions``.

        The doubling (``cat([f, f])``) makes ``x * cos + rotate_half(x) * sin``
        equal the per-pair rotation ``o1 = x1 c - x2 s ; o2 = x2 c + x1 s`` used
        by vLLM's NEOX kernel. The inserted head axis lets the tables broadcast
        against both ``[T, H, D]`` and ``[L, T, H, D]``.
        """
        positions = positions.to(self.device).float()
        angles = positions[:, None] * self.inv_freq[None, :]      # [T, half]
        cos = torch.cat((angles.cos(), angles.cos()), dim=-1)     # [T, rotary_dim]
        sin = torch.cat((angles.sin(), angles.sin()), dim=-1)
        return cos[:, None, :].to(self.dtype), sin[:, None, :].to(self.dtype)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[-1] == self.rotary_dim:
            return x, x.new_empty(x.shape[:-1] + (0,))
        return x[..., : self.rotary_dim], x[..., self.rotary_dim :]

    def apply(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Rotate content -> position space (``x1 c - x2 s ; x2 c + x1 s``)."""
        rot, passthrough = self._split(x)
        rot = rot * cos + _rotate_half(rot) * sin
        return rot if passthrough.numel() == 0 else torch.cat((rot, passthrough), -1)

    def strip(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Inverse rotation: position -> content space (exact, RoPE orthogonal)."""
        rot, passthrough = self._split(x)
        rot = rot * cos - _rotate_half(rot) * sin
        return rot if passthrough.numel() == 0 else torch.cat((rot, passthrough), -1)

    def apply_at(self, x: Tensor, positions: Tensor) -> Tensor:
        cos, sin = self.cos_sin(positions)
        return self.apply(x, cos, sin)

    def strip_at(self, x: Tensor, positions: Tensor) -> Tensor:
        cos, sin = self.cos_sin(positions)
        return self.strip(x, cos, sin)
