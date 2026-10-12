"""Fit the closed-form cross-model KV mapper from captured caches (Phase A / M2).

Consumes the per-sequence caches written by ``KVCaptureConnector`` for the two
family members (run ``scripts/capture_kv.py`` once per model first), strips RoPE
off the keys into content space, runs top-k source selection + per-head ridge,
and serialises the fitted mapper to ``<out>.safetensors`` (+ ``.json`` sidecar).

Example (ROCm; the heavy SPD solve auto-offloads to GPU on LAPACK-less builds)::

    .../python scripts/fit_mapper.py \\
        --source-dir calib/14b --target-dir calib/32b \\
        --out cross_model_kv/artifacts/qwen3_14b_to_32b \\
        --top-k 8 --device cuda \\
        --source-model Qwen/Qwen3-14B --target-model Qwen/Qwen3-32B

Sequences are matched across the two models by their shared ``<seq_hash>``
directory name, so the same calibration prompts must have been captured for both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from cross_model_kv.capture_store import load_sequence, sequence_dirs  # noqa: E402
from cross_model_kv.config import MapperConfig  # noqa: E402
from cross_model_kv.io import save_mapper  # noqa: E402
from cross_model_kv.mapper import RidgeKVMapper  # noqa: E402
from cross_model_kv.ridge import r2_score  # noqa: E402
from cross_model_kv.rope import RopeCodec  # noqa: E402

Tensor = torch.Tensor


def _matched_hashes(source_dir: Path, target_dir: Path) -> list[str]:
    src = {p.name for p in sequence_dirs(source_dir)}
    tgt = {p.name for p in sequence_dirs(target_dir)}
    both = sorted(src & tgt)
    if not both:
        raise SystemExit(
            f"no sequences shared between {source_dir} and {target_dir}; "
            "capture the same calibration prompts for both models first"
        )
    if src - tgt or tgt - src:
        print(
            f"[fit] {len(both)} shared sequences "
            f"({len(src - tgt)} source-only, {len(tgt - src)} target-only ignored)"
        )
    return both


def load_content_space(
    source_dir: Path,
    target_dir: Path,
    src_codec: RopeCodec,
    tgt_codec: RopeCodec,
    device: str,
    max_seqs: int | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Assemble token-concatenated content-space caches for the fit.

    Keys are RoPE-stripped per sequence (positions restart at 0 each prompt);
    values pass through. Returns ``src_k, src_v, tgt_k, tgt_v`` each
    ``[num_layers, num_tokens_total, num_kv_heads, head_dim]``.
    """
    hashes = _matched_hashes(source_dir, target_dir)
    if max_seqs is not None:
        hashes = hashes[:max_seqs]

    src_k_parts, src_v_parts, tgt_k_parts, tgt_v_parts = [], [], [], []
    for h in hashes:
        sk, sv = load_sequence(source_dir / h, device)
        tk, tv = load_sequence(target_dir / h, device)
        n = min(sk.shape[1], tk.shape[1])
        positions = torch.arange(n, device=device)
        src_k_parts.append(src_codec.strip_at(sk[:, :n], positions))
        src_v_parts.append(sv[:, :n])
        tgt_k_parts.append(tgt_codec.strip_at(tk[:, :n], positions))
        tgt_v_parts.append(tv[:, :n])

    cat = lambda parts: torch.cat(parts, dim=1)  # noqa: E731
    return (
        cat(src_k_parts),
        cat(src_v_parts),
        cat(tgt_k_parts),
        cat(tgt_v_parts),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True, help="e.g. calib/14b")
    ap.add_argument("--target-dir", required=True, help="e.g. calib/32b")
    ap.add_argument("--out", required=True, help="artifact stem (no extension)")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--ridge-lambda", type=float, default=1e-2)
    ap.add_argument("--source-rope-theta", type=float, default=1_000_000.0)
    ap.add_argument("--target-rope-theta", type=float, default=1_000_000.0)
    ap.add_argument("--rotary-dim", type=int, default=None,
                    help="defaults to head_dim (full rotary, as in Qwen3)")
    ap.add_argument("--source-model", default="")
    ap.add_argument("--target-model", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seqs", type=int, default=None)
    args = ap.parse_args()

    source_dir, target_dir = Path(args.source_dir), Path(args.target_dir)
    hashes = _matched_hashes(source_dir, target_dir)
    if args.max_seqs is not None:
        hashes = hashes[: args.max_seqs]

    # Probe geometry from the first matched sequence.
    sk0, _ = load_sequence(source_dir / hashes[0], "cpu")
    tk0, _ = load_sequence(target_dir / hashes[0], "cpu")
    num_source_layers, _, num_kv_heads, head_dim = sk0.shape
    num_target_layers = tk0.shape[0]
    print(
        f"[fit] source {num_source_layers}L -> target {num_target_layers}L, "
        f"n_kv={num_kv_heads}, d_h={head_dim}, {len(hashes)} sequences"
    )

    cfg = MapperConfig(
        num_source_layers=num_source_layers,
        num_target_layers=num_target_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        source_rope_theta=args.source_rope_theta,
        target_rope_theta=args.target_rope_theta,
        rotary_dim=args.rotary_dim,
        top_k=args.top_k,
        ridge_lambda=args.ridge_lambda,
        source_model=args.source_model,
        target_model=args.target_model,
    )

    src_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.source_rope_theta,
                          device=args.device)
    tgt_codec = RopeCodec(cfg.head_dim, cfg.rotary_dim, cfg.target_rope_theta,
                          device=args.device)
    src_k, src_v, tgt_k, tgt_v = load_content_space(
        source_dir, target_dir, src_codec, tgt_codec, args.device, args.max_seqs
    )
    print(f"[fit] {src_k.shape[1]} token-level observations "
          f"(feature_dim={cfg.feature_dim})")

    mapper = RidgeKVMapper(cfg).fit(src_k, src_v, tgt_k, tgt_v)

    # In-sample fit quality (paper Table 7 expects ~0.79 K / ~0.65 V at k=8).
    pred_k, pred_v = mapper.transfer(src_k, src_v)
    print(f"[fit] in-sample R^2  K={r2_score(pred_k, tgt_k):.3f}  "
          f"V={r2_score(pred_v, tgt_v):.3f}")
    for t in range(cfg.num_target_layers):
        print(f"  target {t:2d}: sources {mapper.selected[t].tolist()}")

    path = save_mapper(mapper, args.out)
    print(f"[fit] wrote mapper -> {path}")


if __name__ == "__main__":
    main()
