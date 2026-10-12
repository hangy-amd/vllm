"""Closed-form per-head KV mapper (paper Sec 3).

Three components:
    1. Cross-layer source selection (Sec 3.2): per target layer, pick the top-k
       most predictive source layers by head-averaged single-source R^2.
    2. Per-head ridge regression (Sec 3.1): map the concatenated selected source
       KV to each target head, closed form, in content space.
    3. Content-space mapping (Sec 3.3): handled by the caller via :class:`RopeCodec`
       around the key path (strip source RoPE before fit/transfer, re-apply target
       RoPE after transfer). Values carry no positional encoding.

Every target head's map takes the *full* concatenated feature vector (all KV
heads of the selected source layers) as input, which is what enables cross-head
information flow. Solving all heads of a target layer against one shared Gram is
algebraically identical to fitting each head independently, and is the fitting-
time optimisation the paper highlights.

Caches are laid out ``[num_layers, num_tokens, num_kv_heads, head_dim]`` and are
assumed already in content space (RoPE stripped for K).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from cross_model_kv.config import MapperConfig
from cross_model_kv.ridge import fit_device, r2_score, ridge_gram, ridge_solve

Tensor = torch.Tensor


@dataclass
class LayerMap:
    """The fitted map for one target layer: ``Y = X @ weight + bias``.

    ``weight`` is ``[feature_dim, num_kv_heads * head_dim]`` and ``bias`` is
    ``[num_kv_heads * head_dim]``; the per-head maps are its column blocks.
    """

    weight: Tensor
    bias: Tensor


@dataclass
class RidgeKVMapper:
    cfg: MapperConfig
    # Per target layer: selected source-layer indices, and the K / V layer maps.
    selected: list[Tensor] = field(default_factory=list)
    key_maps: list[LayerMap] = field(default_factory=list)
    val_maps: list[LayerMap] = field(default_factory=list)
    # Populated during selection for diagnostics: [num_target, num_source].
    selection_scores: Tensor | None = None

    # ---- component 1: cross-layer source selection --------------------------
    def _select_sources(
        self, src_k: Tensor, src_v: Tensor, tgt_k: Tensor, tgt_v: Tensor
    ) -> None:
        """Rank source layers per target layer by head-averaged single-source R^2.

        The score averages keys and values so K and V share one selection, which
        lets a target head draw complementary information spread across source
        layers. Single-source Grams depend only on the source head, so they are
        formed once and reused across all target layers.
        """
        cfg = self.cfg
        num_src = src_k.shape[0]
        lam = cfg.ridge_lambda

        # Precompute per (source layer, head) centered Grams for K and V.
        grams: dict[str, list[list[tuple[Tensor, Tensor, Tensor]]]] = {}
        for name, src in (("k", src_k), ("v", src_v)):
            per_layer = []
            for s in range(num_src):
                per_head = [
                    ridge_gram(src[s, :, h, :], lam)
                    for h in range(cfg.num_kv_heads)
                ]
                per_layer.append(per_head)
            grams[name] = per_layer

        scores = torch.empty(cfg.num_target_layers, num_src)
        for t in range(cfg.num_target_layers):
            for s in range(num_src):
                acc = 0.0
                for name, tgt in (("k", tgt_k), ("v", tgt_v)):
                    for h in range(cfg.num_kv_heads):
                        gram, xc, x_mean = grams[name][s][h]
                        y = tgt[t, :, h, :]
                        w, b = ridge_solve(gram, xc, x_mean, y)
                        acc += r2_score(xc @ w + x_mean @ w + b, y)
                scores[t, s] = acc / (2 * cfg.num_kv_heads)
            top = torch.topk(scores[t], cfg.top_k).indices.sort().values
            self.selected.append(top)
        self.selection_scores = scores

    # ---- component 2: per-head ridge over the selected source layers --------
    def _gather(self, cache: Tensor, layers: Tensor) -> Tensor:
        """Concatenate all KV heads of the selected source layers -> ``[T, feature_dim]``."""
        num_tokens = cache.shape[1]
        parts = [cache[int(layer)].reshape(num_tokens, -1) for layer in layers]
        return torch.cat(parts, dim=-1)

    def _fit_cache(self, src: Tensor, tgt: Tensor) -> list[LayerMap]:
        cfg = self.cfg
        orig = src.device
        dev = fit_device(orig)
        if dev != orig:
            src, tgt = src.to(dev), tgt.to(dev)
        maps: list[LayerMap] = []
        for t in range(cfg.num_target_layers):
            x = self._gather(src, self.selected[t])
            gram, xc, x_mean = ridge_gram(x, cfg.ridge_lambda)
            # Stack all heads column-wise: one solve for the whole layer.
            y = tgt[t].reshape(tgt.shape[1], -1)     # [T, n_kv * head_dim]
            weight, bias = ridge_solve(gram, xc, x_mean, y)
            maps.append(LayerMap(weight.to(orig), bias.to(orig)))
        return maps

    def fit(
        self, src_k: Tensor, src_v: Tensor, tgt_k: Tensor, tgt_v: Tensor
    ) -> RidgeKVMapper:
        """Fit from paired *content-space* caches (RoPE already stripped for K).

        Heavy Gram formation and the SPD solve run on :func:`fit_device` (an
        accelerator on LAPACK-less ROCm builds); the compact fitted maps are
        returned on the caller's original device.
        """
        self._validate_caches(src_k, src_v, tgt_k, tgt_v)
        orig = src_k.device
        dev = fit_device(orig)
        if dev != orig:
            src_k, src_v, tgt_k, tgt_v = (
                c.to(dev) for c in (src_k, src_v, tgt_k, tgt_v)
            )
        self.selected.clear()
        self._select_sources(src_k, src_v, tgt_k, tgt_v)
        self.key_maps = self._fit_cache(src_k, tgt_k)
        self.val_maps = self._fit_cache(src_v, tgt_v)
        if dev != orig:
            self._to_device(orig)
        return self

    def _to_device(self, device: torch.device) -> RidgeKVMapper:
        """Move the fitted maps and selection metadata onto ``device``."""
        self.selected = [s.to(device) for s in self.selected]
        for m in (*self.key_maps, *self.val_maps):
            m.weight = m.weight.to(device)
            m.bias = m.bias.to(device)
        if self.selection_scores is not None:
            self.selection_scores = self.selection_scores.to(device)
        return self

    # ---- transfer -----------------------------------------------------------
    @staticmethod
    def _apply(x: Tensor, m: LayerMap) -> Tensor:
        """Affine map ``x @ W + b`` on :func:`fit_device`, result on ``x``'s device.

        At serving time ``x`` and the map already share the KV-cache device, so
        this is an in-place no-op; it only offloads the eval-time matmul off a
        LAPACK-less CPU.
        """
        orig = x.device
        dev = fit_device(orig)
        if dev == orig:
            return x @ m.weight + m.bias
        return (x.to(dev) @ m.weight.to(dev) + m.bias.to(dev)).to(orig)

    def _transfer_cache(self, src: Tensor, maps: list[LayerMap]) -> Tensor:
        cfg = self.cfg
        num_tokens = src.shape[1]
        out = src.new_empty(
            cfg.num_target_layers, num_tokens, cfg.num_kv_heads, cfg.head_dim
        )
        for t in range(cfg.num_target_layers):
            x = self._gather(src, self.selected[t]).float()
            mapped = self._apply(x, maps[t])
            out[t] = mapped.reshape(num_tokens, cfg.num_kv_heads, cfg.head_dim).to(
                out.dtype
            )
        return out

    def transfer(self, src_k: Tensor, src_v: Tensor) -> tuple[Tensor, Tensor]:
        """Map source content-space caches to target content-space caches."""
        return (
            self._transfer_cache(src_k, self.key_maps),
            self._transfer_cache(src_v, self.val_maps),
        )

    def transfer_layer(self, src_k: Tensor, src_v: Tensor, target_layer: int
                       ) -> tuple[Tensor, Tensor]:
        """Map to a single target layer -- the serving-time per-layer hook.

        ``src_k`` / ``src_v`` are the full source caches ``[L_s, T, H, D]``; only
        the layers selected for ``target_layer`` are read.
        """
        layers = self.selected[target_layer]
        xk = self._gather(src_k, layers).float()
        xv = self._gather(src_v, layers).float()
        mk, mv = self.key_maps[target_layer], self.val_maps[target_layer]
        shape = (src_k.shape[1], self.cfg.num_kv_heads, self.cfg.head_dim)
        k = self._apply(xk, mk).reshape(shape).to(src_k.dtype)
        v = self._apply(xv, mv).reshape(shape).to(src_v.dtype)
        return k, v

    def _validate_caches(self, *caches: Tensor) -> None:
        cfg = self.cfg
        src_k, src_v, tgt_k, tgt_v = caches
        for name, c, L in (
            ("src_k", src_k, cfg.num_source_layers),
            ("src_v", src_v, cfg.num_source_layers),
            ("tgt_k", tgt_k, cfg.num_target_layers),
            ("tgt_v", tgt_v, cfg.num_target_layers),
        ):
            if c.shape[0] != L or c.shape[2:] != (cfg.num_kv_heads, cfg.head_dim):
                raise ValueError(
                    f"{name} shape {tuple(c.shape)} inconsistent with config "
                    f"(layers={L}, n_kv={cfg.num_kv_heads}, d_h={cfg.head_dim})"
                )
        n = src_k.shape[1]
        if not all(c.shape[1] == n for c in caches):
            raise ValueError("all caches must share the same token count")
        if n < cfg.feature_dim:
            raise ValueError(
                f"under-determined: N={n} calibration tokens < feature_dim="
                f"{cfg.feature_dim}; need more tokens or smaller top_k"
            )
