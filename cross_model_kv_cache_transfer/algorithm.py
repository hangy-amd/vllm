"""Cross-model KV cache transfer -- closed-form ridge mapper.

Illustrative reference implementation of

    Heo et al., "Cross-Model KV Cache Transfer in LLM Families:
    A Closed-Form Linear Mapping for Prefill Reuse" (arXiv:2608.03893).

Idea: when a serving stack swaps between two members of the same model family,
the receiver normally re-runs *prefill* to populate its KV cache. The paper shows
the cross-model KV relationship is largely *linear*, so the receiver can instead
reuse the sender's already-prefilled cache through a cheap per-head linear map --
skipping prefill entirely.

The mapper has three components, each fit with a closed-form solve (no gradients):

    1. Cross-layer source selection (Sec 3.2)
       For each *target* layer, pick the top-k most predictive *source* layers by
       head-averaged R^2, and concatenate their KV as the regression input. This
       is the single largest contributor to quality in the paper's ablation.

    2. Content-space mapping / RoPE factoring (Sec 3.3)
       Keys are stored with a position-dependent RoPE rotation. We strip it before
       fitting, so the map lives in a *position-free* content space and transfers
       across context lengths. After mapping we re-apply the target's RoPE. Values
       carry no positional encoding and are mapped directly.

    3. Per-head ridge regression (Sec 3.1)
       For each (target layer, head) and each of K / V, solve the ridge system
       in closed form:   W* = (X^T X + lambda I)^-1 X^T Y.
       Nothing is shared across heads or between K and V.

This file uses synthetic-but-structured caches so the whole pipeline is runnable
and its recovery is visible. In a real deployment `source_*` / `target_*` come
from running prefill on the *same* calibration text through the two models.
"""

from dataclasses import dataclass, field

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # Architecture of the two family members. "Matched-KV" (the paper's setting)
    # means source and target agree on KV-head count and per-head dimension, even
    # though depth and parameter count differ.
    num_source_layers: int = 8
    num_target_layers: int = 12
    num_kv_heads: int = 8          # n_kv, matched across source/target
    head_dim: int = 128            # d_h, matched across source/target

    # RoPE base ("theta"). Family members frequently differ here -- content-space
    # mapping is exactly what makes the fit robust to that.
    source_rope_base: float = 10_000.0
    target_rope_base: float = 1_000_000.0

    # Mapper hyper-parameters.
    top_k: int = 4                 # source layers concatenated per target layer
    ridge_lambda: float = 1e-2     # Tikhonov term, paper's default

    # Calibration vs. evaluation token budgets. Calibration must over-determine
    # the per-head system: N_calib > top_k * n_kv * d_h. Evaluation deliberately
    # uses *later, unseen positions* to exercise length generalization.
    num_calib_tokens: int = 6144
    num_eval_tokens: int = 1024

    # Ground-truth synthetic relationship (unknown to the mapper).
    true_sources_per_target: int = 3   # how many source layers each target mixes
    noise_std: float = 0.05

    seed: int = 0

    @property
    def source_feature_dim(self) -> int:
        # Width of one source layer's flattened KV: all heads concatenated.
        return self.num_kv_heads * self.head_dim


# --------------------------------------------------------------------------- #
# RoPE helpers (content <-> position space)
# --------------------------------------------------------------------------- #
def rope_cos_sin(positions: Tensor, head_dim: int, base: float) -> tuple[Tensor, Tensor]:
    """Cosine / sine tables for RoPE, shaped ``[T, head_dim]`` (Llama layout)."""
    half = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, device=positions.device).float() / half))
    angles = positions.float()[:, None] * inv_freq[None, :]   # [T, half]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)     # [T, head_dim]
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate content keys into position space. ``x`` is ``[..., T, H, D]``."""
    cos, sin = cos[:, None, :], sin[:, None, :]               # broadcast over heads
    return x * cos + _rotate_half(x) * sin


def strip_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Inverse rotation: position space -> content space (exact, RoPE is orthogonal)."""
    cos, sin = cos[:, None, :], sin[:, None, :]
    return x * cos - _rotate_half(x) * sin


# --------------------------------------------------------------------------- #
# Core linear algebra: closed-form ridge + diagnostics
# --------------------------------------------------------------------------- #
def ridge_fit(x: Tensor, y: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    """Closed-form ridge regression  y ~= x @ W + b.

    Centering both sides lets us solve for the slope alone and recover the bias
    as ``b = mean(y) - mean(x) @ W`` (paper Eq. 4). Computed in float32 for
    conditioning, since the concatenated source layers are correlated and make
    ``X^T X`` near-singular -- the whole reason for the Tikhonov term.
    """
    x, y = x.float(), y.float()
    x_mean, y_mean = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
    xc, yc = x - x_mean, y - y_mean

    gram = xc.T @ xc
    gram.diagonal().add_(lam)                                 # (X^T X + lambda I)
    weight = torch.linalg.solve(gram, xc.T @ yc)
    bias = y_mean - x_mean @ weight
    return weight, bias.squeeze(0)


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Multivariate coefficient of determination, aggregated over all elements."""
    pred, target = pred.float(), target.float()
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean(0, keepdim=True)).pow(2).sum()  # hangy: target.mean(0, keepdim=True)是沿第0维求平均
    return (1.0 - ss_res / ss_tot).item()


# --------------------------------------------------------------------------- #
# The mapper
# --------------------------------------------------------------------------- #
@dataclass
class RidgeKVMapper:
    """Per-head closed-form KV mapper operating in content space.

    ``fit`` and ``transfer`` are pure linear algebra: the caller is responsible
    for stripping/re-applying RoPE around the key path (see ``main``). Caches are
    laid out ``[num_layers, num_tokens, num_kv_heads, head_dim]``.
    """

    cfg: Config
    # For each target layer: the selected source-layer indices and, per head,
    # the (W, b) pair for keys and for values.
    selected: list[Tensor] = field(default_factory=list)
    key_maps: list[list[tuple[Tensor, Tensor]]] = field(default_factory=list)
    val_maps: list[list[tuple[Tensor, Tensor]]] = field(default_factory=list)

    # ---- component 1: cross-layer source selection --------------------------
    def _select_sources(
        self, src_k: Tensor, src_v: Tensor, tgt_k: Tensor, tgt_v: Tensor
    ) -> None:
        """Rank source layers per target layer by head-averaged single-source R^2.

        The score averages keys and values so K and V share one selection, which
        lets a head draw on complementary information spread across source layers.
        """
        num_src = src_k.shape[0]
        for tgt_layer in range(self.cfg.num_target_layers):  # hangy: for each target layer
            scores = torch.empty(num_src)
            for src_layer in range(num_src):  # hangy: for each source layer
                r2_heads = []
                for cache_src, cache_tgt in ((src_k, tgt_k), (src_v, tgt_v)):
                    for head in range(self.cfg.num_kv_heads):  # hangy: for k and v and each head
                        x = cache_src[src_layer, :, head, :]
                        y = cache_tgt[tgt_layer, :, head, :]
                        w, b = ridge_fit(x, y, self.cfg.ridge_lambda)
                        r2_heads.append(r2_score(x.float() @ w + b, y))
                scores[src_layer] = sum(r2_heads) / len(r2_heads)
            top = torch.topk(scores, self.cfg.top_k).indices.sort().values
            self.selected.append(top)

    # ---- component 3: per-head ridge over the selected source layers --------
    def _gather(self, cache: Tensor, layers: Tensor) -> Tensor:
        """Concatenate all KV heads of the selected source layers: ``[T, k*n_kv*d_h]``."""
        num_tokens = cache.shape[1]
        parts = [cache[layer].reshape(num_tokens, -1) for layer in layers]
        return torch.cat(parts, dim=-1)

    def _fit_cache(
        self, src: Tensor, tgt: Tensor
    ) -> list[list[tuple[Tensor, Tensor]]]:
        maps: list[list[tuple[Tensor, Tensor]]] = []
        for tgt_layer in range(self.cfg.num_target_layers):
            x = self._gather(src, self.selected[tgt_layer])
            head_maps = [
                ridge_fit(x, tgt[tgt_layer, :, head, :], self.cfg.ridge_lambda)
                for head in range(self.cfg.num_kv_heads)
            ]
            maps.append(head_maps)
        return maps

    def fit(self, src_k: Tensor, src_v: Tensor, tgt_k: Tensor, tgt_v: Tensor) -> "RidgeKVMapper":
        """Fit the mapper from paired *content-space* caches (RoPE already stripped for K)."""
        self.selected.clear()
        self._select_sources(src_k, src_v, tgt_k, tgt_v)
        self.key_maps = self._fit_cache(src_k, tgt_k)
        self.val_maps = self._fit_cache(src_v, tgt_v)
        return self

    def _transfer_cache(
        self, src: Tensor, maps: list[list[tuple[Tensor, Tensor]]]
    ) -> Tensor:
        num_tokens = src.shape[1]
        out = src.new_empty(
            self.cfg.num_target_layers,
            num_tokens,
            self.cfg.num_kv_heads,
            self.cfg.head_dim,
        )
        for tgt_layer in range(self.cfg.num_target_layers):
            x = self._gather(src, self.selected[tgt_layer]).float()
            for head, (w, b) in enumerate(maps[tgt_layer]):
                out[tgt_layer, :, head, :] = (x @ w + b).to(out.dtype)
        return out

    def transfer(self, src_k: Tensor, src_v: Tensor) -> tuple[Tensor, Tensor]:
        """Map a source content-space cache to the target's content-space cache."""
        return self._transfer_cache(src_k, self.key_maps), self._transfer_cache(src_v, self.val_maps)


# --------------------------------------------------------------------------- #
# Attention-output cosine (the paper's cross-pair retention predictor, Sec 4.5)
# --------------------------------------------------------------------------- #
def attention_output(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Causal scaled-dot-product attention per head. All inputs ``[T, H, D]``."""
    scale = 1.0 / (q.shape[-1] ** 0.5)
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * scale
    mask = torch.triu(torch.ones_like(scores[0], dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return torch.einsum("hts,shd->thd", attn, v.float())


def attention_cosine(q: Tensor, k_true: Tensor, v_true: Tensor, k_pred: Tensor, v_pred: Tensor) -> float:
    """Mean cosine similarity between attention outputs from true vs. mapped KV.

    The paper finds this tracks downstream retention far better than raw R^2:
    what matters is whether the *attention output* the target would compute is
    preserved, not per-channel reconstruction error.
    """
    out_true = attention_output(q, k_true, v_true).flatten(start_dim=1)
    out_pred = attention_output(q, k_pred, v_pred).flatten(start_dim=1)
    return torch.cosine_similarity(out_true, out_pred, dim=-1).mean().item()


# --------------------------------------------------------------------------- #
# Synthetic paired caches with genuine cross-layer linear structure
# --------------------------------------------------------------------------- #
@dataclass
class GroundTruth:
    """The (hidden) linear relationship the mapper must recover."""
 
    key_sources: list[Tensor]      # per target layer: source layer indices
    val_sources: list[Tensor]
    key_mix: list[Tensor]          # per target layer: [k*n_kv*d_h, n_kv*d_h]
    val_mix: list[Tensor]


def build_ground_truth(cfg: Config, gen: torch.Generator, device: str) -> GroundTruth:
    feat = cfg.source_feature_dim  # hangy: num_kv_head * kv_head_dim
    k = cfg.true_sources_per_target
    scale = (k * feat) ** -0.5                       # keep target variance ~ O(1)

    def draw_mix() -> list[Tensor]:
        return [
            torch.randn(k * feat, feat, generator=gen, device=device) * scale
            for _ in range(cfg.num_target_layers)
        ]

    # Keys and values of a target layer draw on the *same* informative source
    # layers (only the linear mixing differs), matching how the paper shares one
    # top-k selection across K and V.
    sources = [
        torch.randperm(cfg.num_source_layers, generator=gen, device=device)[:k].sort().values
        for _ in range(cfg.num_target_layers)
    ]
    return GroundTruth(sources, sources, draw_mix(), draw_mix())  # hangy: GroundTruth是指ground_truth mapper


def sample_caches(
    cfg: Config, gt: GroundTruth, positions: Tensor, gen: torch.Generator, device: str
) -> dict[str, Tensor]:
    """Draw source content, propagate through the ground-truth mix, apply RoPE.

    Returns the *stored* caches a serving stack would actually hold: keys carry
    RoPE, values do not. The mapper only ever sees these.
    """
    num_tokens = positions.shape[0]
    ls, lt = cfg.num_source_layers, cfg.num_target_layers
    n_kv, d_h = cfg.num_kv_heads, cfg.head_dim
    shape = (num_tokens, n_kv, d_h)

    src_k_c = torch.randn(ls, *shape, generator=gen, device=device)
    src_v = torch.randn(ls, *shape, generator=gen, device=device)  # hangy: get content-space kv

    def mix(src: Tensor, sources: list[Tensor], mixes: list[Tensor]) -> Tensor:
        out = torch.empty(lt, *shape, device=device)
        for layer in range(lt):
            x = torch.cat([src[s].reshape(num_tokens, -1) for s in sources[layer]], dim=-1)
            noise = torch.randn(num_tokens, n_kv * d_h, generator=gen, device=device)
            out[layer] = (x @ mixes[layer] + cfg.noise_std * noise).reshape(shape)
        return out

    tgt_k_c = mix(src_k_c, gt.key_sources, gt.key_mix)  # hangy: apply ground_truth mapper to content-space
    tgt_v = mix(src_v, gt.val_sources, gt.val_mix)

    src_cos, src_sin = rope_cos_sin(positions, d_h, cfg.source_rope_base)
    tgt_cos, tgt_sin = rope_cos_sin(positions, d_h, cfg.target_rope_base)
    return {
        "src_k_rope": apply_rope(src_k_c, src_cos, src_sin),
        "src_v": src_v,
        "tgt_k_rope": apply_rope(tgt_k_c, tgt_cos, tgt_sin),
        "tgt_v": tgt_v,
        "positions": positions,
    }


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    d_h = cfg.head_dim
    gt = build_ground_truth(cfg, gen, device)

    calib_pos = torch.arange(cfg.num_calib_tokens, device=device)
    # Evaluation uses *later* positions than calibration -- unseen by the fit.
    eval_pos = torch.arange(
        cfg.num_calib_tokens, cfg.num_calib_tokens + cfg.num_eval_tokens, device=device
    )  # hangy: calib和eval的pos
    calib = sample_caches(cfg, gt, calib_pos, gen, device)
    ev = sample_caches(cfg, gt, eval_pos, gen, device)

    # RoPE tables for the two splits (keys live in position space when stored).
    src_cos_cal, src_sin_cal = rope_cos_sin(calib_pos, d_h, cfg.source_rope_base)
    tgt_cos_cal, tgt_sin_cal = rope_cos_sin(calib_pos, d_h, cfg.target_rope_base)
    src_cos_ev, src_sin_ev = rope_cos_sin(eval_pos, d_h, cfg.source_rope_base)
    tgt_cos_ev, tgt_sin_ev = rope_cos_sin(eval_pos, d_h, cfg.target_rope_base)

    # --- Content-space pipeline (the paper's method) ------------------------
    # Strip RoPE off the stored keys to fit position-free maps.
    src_k_cal_c = strip_rope(calib["src_k_rope"], src_cos_cal, src_sin_cal)
    tgt_k_cal_c = strip_rope(calib["tgt_k_rope"], tgt_cos_cal, tgt_sin_cal)

    mapper = RidgeKVMapper(cfg).fit(
        src_k_cal_c, calib["src_v"], tgt_k_cal_c, calib["tgt_v"]
    )

    # Transfer: strip source RoPE, map, re-apply *target* RoPE.
    src_k_ev_c = strip_rope(ev["src_k_rope"], src_cos_ev, src_sin_ev)
    pred_k_ev_c, pred_v_ev = mapper.transfer(src_k_ev_c, ev["src_v"])
    pred_k_ev_rope = apply_rope(pred_k_ev_c, tgt_cos_ev, tgt_sin_ev)
    tgt_k_ev_c = strip_rope(ev["tgt_k_rope"], tgt_cos_ev, tgt_sin_ev)

    r2_k = r2_score(pred_k_ev_c, tgt_k_ev_c)
    r2_v = r2_score(pred_v_ev, ev["tgt_v"])

    # Attention-output cosine with random target queries (RoPE-encoded).
    q_c = torch.randn(cfg.num_eval_tokens, cfg.num_kv_heads, d_h, generator=gen, device=device)
    q_rope = apply_rope(q_c, tgt_cos_ev, tgt_sin_ev)
    cos_layers = [
        attention_cosine(
            q_rope,
            ev["tgt_k_rope"][l], ev["tgt_v"][l],
            pred_k_ev_rope[l], pred_v_ev[l],
        )
        for l in range(cfg.num_target_layers)
    ]
    cos_mean = sum(cos_layers) / len(cos_layers)

    # --- Ablation: RoPE-coupled fit (no content-space) ----------------------
    # Fit keys directly on stored (rotated) caches, reusing the same selection.
    # Because the map is tied to calibration positions, it degrades at the later
    # evaluation positions -- the paper's motivation for content-space mapping.
    coupled = RidgeKVMapper(cfg)
    coupled.selected = mapper.selected
    coupled.key_maps = coupled._fit_cache(calib["src_k_rope"], calib["tgt_k_rope"])
    pred_k_coupled = coupled._transfer_cache(ev["src_k_rope"], coupled.key_maps)
    r2_k_coupled = r2_score(pred_k_coupled, ev["tgt_k_rope"])

    # --- Report -------------------------------------------------------------
    feat = cfg.top_k * cfg.source_feature_dim
    print(f"device={device}  matched-KV: n_kv={cfg.num_kv_heads}, d_h={d_h}")
    print(f"source layers={cfg.num_source_layers} -> target layers={cfg.num_target_layers}")
    print(f"per-head ridge input dim = top_k*n_kv*d_h = {feat}  (N_calib={cfg.num_calib_tokens})")
    print("selected source layers per target layer:")
    for l, sel in enumerate(mapper.selected):
        truth = gt.key_sources[l].tolist()
        print(f"  target {l:2d}: picked {sel.tolist()}   (K ground-truth {truth})")
    print("--- transfer quality on unseen evaluation positions ---")
    print(f"  keys   R^2 (content-space) : {r2_k:6.3f}")
    print(f"  values R^2                 : {r2_v:6.3f}")
    print(f"  keys   R^2 (RoPE-coupled)  : {r2_k_coupled:6.3f}   <- degrades w/o RoPE stripping")
    print(f"  attention-output cosine    : {cos_mean:6.3f}")


if __name__ == "__main__":
    main()
