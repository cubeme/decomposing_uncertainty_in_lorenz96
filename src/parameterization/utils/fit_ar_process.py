"""Fit autoregressive processes to parameterization residuals."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.polynomial.polynomial as poly
from absl import logging


# -------------------------
# Stability checks and projection for AR(p) coefficients
# -------------------------
def is_stable_ar(rho: np.ndarray, tol: float = 1e-6) -> bool:
    """
    AR(p) is stationary iff all roots of:
        1 - rho_1 z - rho_2 z^2 - ... - rho_p z^p = 0
    satisfy |root| > 1.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size == 0:
        return True

    # polynomial in descending powers for np.roots: [-rho_p, ..., -rho_1, 1]
    coeffs = np.r_[-rho[::-1], 1.0]
    roots = np.roots(coeffs)
    if not np.all(np.isfinite(roots)):
        return False
    return bool(np.all(np.abs(roots) > 1.0 + tol))


def project_to_stable_ar(
    rho: np.ndarray,
    max_iter: int = 60,
    shrink: float = 0.98,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """
    Simple guardrail: if unstable, scale rho <- scale * rho until stable.
    Preserves relative shape; reduces magnitude.
    """
    rho = np.asarray(rho, dtype=float).reshape(-1)
    if rho.size == 0:
        return rho, True
    if is_stable_ar(rho, tol=tol):
        return rho, True

    scale = 1.0
    for _ in range(max_iter):
        scale *= shrink
        cand = rho * scale
        if is_stable_ar(cand, tol=tol):
            return cand, True
    return rho * (shrink**max_iter), False


# -------------------------
# Core AR(p) fitting (pooled across K)
# -------------------------
def _as_2d_time_series(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected (T,) or (T,K). Got shape {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError("Need at least 2 time steps.")
    return x


def torch_to_numpy_2d(x) -> np.ndarray:
    """
    Accepts torch.Tensor or np.ndarray; returns np.ndarray (T,K) or (T,1).
    """
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return _as_2d_time_series(np.asarray(x))


def fit_ar_p_ls_pooled(
    series: np.ndarray,
    p: int,
    center: bool = True,
    clamp: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    """
    Pooled least squares for AR(p):
        s_t = sum_{i=1..p} rho_i s_{t-i} + e_t
    Returns:
        rho (p,), sigma_i = std(e_t) (scalar)
    """
    s = _as_2d_time_series(series).astype(float, copy=False)
    T, K = s.shape
    if p <= 0:
        return np.zeros((0,), dtype=float), float(np.std(s))

    if T <= p:
        raise ValueError(f"Need T > p. Got T={T}, p={p}.")

    if center:
        s = s - s.mean(axis=0, keepdims=True)

    # Build y: s[p:]  (T-p, K)
    # Build X blocks: [s[p-1:T-1], s[p-2:T-2], ..., s[0:T-p]] (each (T-p, K))
    y = s[p:]  # (T-p, K)
    X_blocks = [s[p - i : T - i] for i in range(1, p + 1)]  # list of (T-p, K)
    X = np.stack(X_blocks, axis=1)  # (T-p, p, K)

    # Pool across K as extra samples: (T-p)*K samples of p predictors
    X2 = np.transpose(X, (0, 2, 1)).reshape(-1, p)  # ((T-p)*K, p)
    y2 = y.reshape(-1, 1)  # ((T-p)*K, 1)

    # Least squares
    rho, *_ = np.linalg.lstsq(X2, y2, rcond=None)
    rho = rho.reshape(-1)  # (p,)

    # Optional per-coefficient clamp (does NOT guarantee stationarity for p>1)
    if clamp is not None and clamp > 0:
        rho = np.clip(rho, -1.0 + clamp, 1.0 - clamp)

    # Innovation std
    yhat2 = X2 @ rho.reshape(-1, 1)
    e2 = y2 - yhat2
    sigma_i = float(np.sqrt(np.mean(e2**2)))
    sigma_i = float(max(sigma_i, 1e-12))
    return rho, sigma_i


def fit_ar_p_yw_pooled(
    series: np.ndarray,
    p: int,
    center: bool = True,
    ridge: float = 1e-8,
) -> Tuple[np.ndarray, float]:
    """
    Pooled Yule–Walker for AR(p), using pooled autocovariances gamma(lag).
    Returns:
        rho (p,), sigma_i innovation std (scalar), via:
            sigma_i^2 = gamma(0) - rho^T r
    where r = [gamma(1),...,gamma(p)].
    """
    s = _as_2d_time_series(series).astype(float, copy=False)
    T, K = s.shape
    if p <= 0:
        return np.zeros((0,), dtype=float), float(np.std(s))

    if T <= p:
        raise ValueError(f"Need T > p. Got T={T}, p={p}.")

    if center:
        s = s - s.mean(axis=0, keepdims=True)

    # gamma[0..p]
    gamma = np.empty((p + 1,), dtype=float)
    for lag in range(p + 1):
        a = s[lag:]  # (T-lag, K)
        b = s[: T - lag]  # (T-lag, K)
        gamma[lag] = float(np.mean(a * b))

    # Toeplitz R from gamma[0..p-1]
    R = np.empty((p, p), dtype=float)
    for i in range(p):
        for j in range(p):
            R[i, j] = gamma[abs(i - j)]
    R += float(ridge) * np.eye(p)

    r = gamma[1:]  # (p,)
    rho = np.linalg.solve(R, r)  # (p,)

    # Innovation variance estimate
    sigma2 = float(gamma[0] - rho @ r)
    sigma2 = max(sigma2, 1e-12)
    sigma_i = float(np.sqrt(sigma2))
    return rho, sigma_i


def fit_ar_p_pooled(
    series: np.ndarray,
    p: int,
    method: str,
    center: bool = True,
    clamp: float = 1e-6,  # for LS only
    ridge: float = 1e-8,  # for YW only
    enforce_stability: bool = True,
    stability_tol: float = 1e-6,
    shrink: float = 0.98,
    max_iter: int = 60,
) -> Tuple[np.ndarray, float]:
    """
    One entry point you can use for BOTH baselines (residuals) and flow latents (z).
    Produces comparable (rho, sigma_i) where sigma_i is the INNOVATION std.

    - series: (T,K) or (T,)
    - p: AR order
    - method: "least_squares" or "yule_walker"
    """
    if method == "least_squares":
        rho, sigma_i = fit_ar_p_ls_pooled(series, p, center=center, clamp=clamp)
    elif method == "yule_walker":
        rho, sigma_i = fit_ar_p_yw_pooled(series, p, center=center, ridge=ridge)
    else:
        raise ValueError(
            f"Unknown method={method!r}. Use 'least_squares' or 'yule_walker'."
        )

    stable_before = is_stable_ar(rho, tol=stability_tol)
    stable_after = stable_before

    if enforce_stability and p > 0 and not stable_before:
        rho2, stable_after = project_to_stable_ar(
            rho, max_iter=max_iter, shrink=shrink, tol=stability_tol
        )
        rho = rho2

        # Recompute sigma_i after projection in a consistent way:
        # use residual MSE under the projected rho (pooled).
        s = _as_2d_time_series(series).astype(float, copy=False)
        if center:
            s = s - s.mean(axis=0, keepdims=True)
        T, K = s.shape
        y = s[p:]  # (T-p, K)
        preds = np.zeros_like(y)
        for i in range(1, p + 1):
            preds += rho[i - 1] * s[p - i : T - i]
        e = y - preds
        sigma_i = float(np.sqrt(np.mean(e**2)))
        sigma_i = float(max(sigma_i, 1e-12))

    if not stable_after:
        logging.warning(
            "AR(p) fit is unstable even after projection. "
            "Consider reducing clamp/shrink or increasing max_iter."
        )

    return np.asarray(rho, dtype=float), float(sigma_i)


# -------------------------
# Wrappers for specific use cases (baseline poly residuals, flow latents)
# -------------------------
def fit_baseline_poly_ar(
    x: np.ndarray,
    u: np.ndarray,
    poly_coefs: np.ndarray,
    p: int = 1,
    method: str = "least_squares",
    enforce_stability: bool = True,
) -> Tuple[float | np.ndarray, float]:
    """
    Baseline:
        residual r_t = u_t - m_poly(x_t)
        fit AR(p) on r_t (pooled across K), return (phi or rho, sigma_i)

    Returns:
      - if p==1: (phi: float, sigma_i: float)
      - else: (rho: np.ndarray shape (p,), sigma_i: float)
    """
    x = _as_2d_time_series(x)
    u = _as_2d_time_series(u)
    if x.shape != u.shape:
        raise ValueError(f"x and u must have same shape. Got {x.shape} vs {u.shape}.")
    if poly_coefs.ndim != 1:
        raise ValueError(f"Expected 1d poly_coefs. Got shape {poly_coefs.shape}.")

    # compute residuals r = u - m_poly(x), then fit AR(p) on r (pooled across K)
    m = poly.polyval(x, poly_coefs)
    r = u - m

    rho, sigma_i = fit_ar_p_pooled(
        r, p=p, method=method, enforce_stability=enforce_stability
    )
    if p == 1:
        return float(rho[0]), float(sigma_i)
    return rho, float(sigma_i)


def fit_flow_latent_ar(
    z: np.ndarray,
    p: int = 1,
    method: str = "least_squares",
    enforce_stability: bool = True,
    burn_in: int = 0,
) -> Tuple[float | np.ndarray, float]:
    """
    Flow latent:
        given inferred latent z_t (T,K), fit AR(p) on z (pooled across K)

    Returns:
      - if p==1: (phi: float, sigma_i: float)
      - else: (rho: np.ndarray shape (p,), sigma_i: float)
    """
    z = torch_to_numpy_2d(z)
    if burn_in > 0:
        if z.shape[0] <= burn_in + p:
            raise ValueError("burn_in too large for available T.")
        z = z[burn_in:]

    rho, sigma_i = fit_ar_p_pooled(
        z, p=p, method=method, enforce_stability=enforce_stability
    )
    if p == 1:
        return float(rho[0]), float(sigma_i)
    return rho, float(sigma_i)
