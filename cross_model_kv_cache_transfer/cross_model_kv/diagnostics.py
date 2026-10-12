"""Attention-output cosine (paper Sec 4.5).

The paper finds this tracks downstream retention far better than raw R^2: what
matters is whether the *attention output* the target would compute is preserved,
not per-channel reconstruction error. Keys/queries are expected post-RoPE here.
"""

from __future__ import annotations

import torch

Tensor = torch.Tensor


def attention_output(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal scaled-dot-product attention per head. Inputs ``[T, H, D]``.

    Supports grouped-query attention: if ``q`` has more heads than ``k``/``v``,
    query heads are mapped to KV heads in contiguous groups.
    """
    t_q, h_q, d = q.shape
    h_kv = k.shape[1]
    if h_q != h_kv:
        if h_q % h_kv != 0:
            raise ValueError(f"query heads {h_q} not a multiple of kv heads {h_kv}")
        rep = h_q // h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = 1.0 / (d**0.5)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    mask = torch.triu(torch.ones_like(scores[0], dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return torch.einsum("hts,shd->thd", attn, v.float())


def attention_cosine(
    q: Tensor, k_true: Tensor, v_true: Tensor, k_pred: Tensor, v_pred: Tensor
) -> float:
    """Mean cosine similarity between attention outputs from true vs mapped KV."""
    out_true = attention_output(q, k_true, v_true).flatten(start_dim=1)
    out_pred = attention_output(q, k_pred, v_pred).flatten(start_dim=1)
    return torch.cosine_similarity(out_true, out_pred, dim=-1).mean().item()
