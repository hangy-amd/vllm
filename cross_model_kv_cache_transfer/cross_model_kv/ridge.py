"""Closed-form ridge regression and R^2 diagnostics (paper Sec 3.1)."""

from __future__ import annotations

import torch

Tensor = torch.Tensor

# ROCm/HIP torch builds are frequently compiled without CPU LAPACK, so a CPU
# ``torch.linalg.solve`` raises at runtime. The Gram solve is cheap to run on
# the accelerator (rocSOLVER), which is where real fitting happens anyway, so we
# transparently offload SPD solves when the host lacks LAPACK.
_CPU_LINALG_OK: bool | None = None


def _cpu_linalg_available() -> bool:
    global _CPU_LINALG_OK
    if _CPU_LINALG_OK is None:
        try:
            torch.linalg.solve(torch.eye(2), torch.ones(2, 1))
            _CPU_LINALG_OK = True
        except RuntimeError:
            _CPU_LINALG_OK = False
    return _CPU_LINALG_OK


def fit_device(device: torch.device) -> torch.device:
    """Device on which to run the heavy Gram/solve work for inputs on ``device``.

    Keeps work in place when ``device`` can factorise (any accelerator, or a
    LAPACK-enabled CPU), otherwise steers CPU inputs to an accelerator so the
    ridge fit runs at all on LAPACK-less ROCm/HIP builds.
    """
    if device.type != "cpu" or _cpu_linalg_available():
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return device


def _spd_solve(gram: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``gram @ X = rhs`` for symmetric positive-definite ``gram``.

    Falls back to an available accelerator when the host torch build has no CPU
    LAPACK, returning the result on ``rhs``'s original device.
    """
    if gram.is_cuda or _cpu_linalg_available():
        return torch.linalg.solve(gram, rhs)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This torch build lacks CPU LAPACK and no accelerator is available "
            "for the ridge solve; install a LAPACK-enabled torch or run on GPU."
        )
    dev = rhs.device
    return torch.linalg.solve(gram.cuda(), rhs.cuda()).to(dev)


def ridge_gram(x: Tensor, lam: float) -> tuple[Tensor, Tensor, Tensor]:
    """Centered Gram matrix ``(X^T X + lambda I)`` and centering statistics.

    Computed in float32 for conditioning: the top-k selected source layers are
    correlated by construction, so ``X^T X`` is near-singular and the Tikhonov
    term stabilises the inverse (paper Eq. 4).

    Args:
        x: Design matrix ``[N, D]``.
        lam: Tikhonov (ridge) coefficient.

    Returns:
        gram: ``[D, D]`` regularised Gram of the centered design.
        xc: Centered design ``[N, D]`` (float32).
        x_mean: ``[1, D]`` column means used for centering.
    """
    x = x.float()
    x_mean = x.mean(0, keepdim=True)
    xc = x - x_mean
    gram = xc.T @ xc
    gram.diagonal().add_(lam)
    return gram, xc, x_mean


def ridge_solve(
    gram: Tensor, xc: Tensor, x_mean: Tensor, y: Tensor
) -> tuple[Tensor, Tensor]:
    """Solve ``y ~= x @ W + b`` given a precomputed centered Gram.

    Sharing ``gram``/``xc`` across every target head of a layer is the key
    fitting-time optimisation: forming ``X^T X`` (``O(N D^2)``) dominates wall
    time and is independent of ``Y`` (paper Sec 3.1). ``Y`` may stack multiple
    heads column-wise and is solved in a single call.

    Args:
        gram: ``[D, D]`` regularised Gram from :func:`ridge_gram`.
        xc: Centered design ``[N, D]``.
        x_mean: ``[1, D]`` column means from :func:`ridge_gram`.
        y: Response ``[N, M]`` (M may span several heads).

    Returns:
        weight: ``[D, M]`` slope.
        bias: ``[M]`` intercept, ``mean(y) - mean(x) @ W``.
    """
    y = y.float()
    y_mean = y.mean(0, keepdim=True)
    weight = _spd_solve(gram, xc.T @ (y - y_mean))
    bias = (y_mean - x_mean @ weight).squeeze(0)
    return weight, bias


def ridge_fit(x: Tensor, y: Tensor, lam: float) -> tuple[Tensor, Tensor]:
    """Closed-form ridge regression ``y ~= x @ W + b`` (convenience wrapper)."""
    gram, xc, x_mean = ridge_gram(x, lam)
    return ridge_solve(gram, xc, x_mean, y)


def r2_score(pred: Tensor, target: Tensor) -> float:
    """Multivariate coefficient of determination, aggregated over all elements."""
    pred, target = pred.float(), target.float()
    ss_res = (target - pred).pow(2).sum()
    ss_tot = (target - target.mean(0, keepdim=True)).pow(2).sum()
    return (1.0 - ss_res / ss_tot).item()
