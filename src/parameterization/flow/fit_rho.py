"""Estimate autoregressive parameters for flow latent variables."""

import numpy as np
import torch
import torch.nn as nn

from parameterization.flow.training.data import (  #
    build_condition,
    ensure_forcing_1d,
    ensure_time_series_2d,
    full_step_stride,
)
from parameterization.utils.fit_ar_process import fit_flow_latent_ar


@torch.no_grad()
def infer_z(
    flow: nn.Module,
    x: torch.Tensor,
    u: torch.Tensor,
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Infer latent noise z_t from observed (x_t, u_t) using the trained inverse flow.

    IMPORTANT: Uses the exact same condition builder as training:
      cond has length T (with start padding in the lagged features)
      u has length T
    """
    if x.dim() not in (1, 2) or u.dim() not in (1, 2):
        raise ValueError(
            f"Expected x,u with shape (T,) or (T,K). Got x={x.shape}, u={u.shape}."
        )

    flow.eval()
    device = next(flow.parameters()).device

    x = ensure_time_series_2d(x, name="x").to(device)
    u = ensure_time_series_2d(u, name="u").to(device)

    if x.shape[0] != u.shape[0]:
        raise ValueError(
            f"x and u must have same length T. Got {x.shape[0]} vs {u.shape[0]}."
        )

    F_dev = None
    if F is not None:
        F_dev = torch.as_tensor(F, device=device, dtype=x.dtype)
        F_dev = ensure_forcing_1d(F_dev, x.shape[0])

    cond = build_condition(x, delta_t=int(delta_t), si=si, dt_full=dt_full, F=F_dev)

    z, _ = flow.f_inv(u, cond)
    return z


@torch.no_grad()
def fit_rho_sigma_p_from_data(
    flow: nn.Module,
    x: torch.Tensor,
    u: torch.Tensor,
    p: int,
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F: torch.Tensor | None = None,
    enforce_stability: bool = True,
    method: str = "least_squares",
) -> tuple[float | np.ndarray, float]:
    """
    Infer z from the trained flow, then fit AR(p) rho via Least-Sqaures or
    Yule–Walker and sigma from residuals.

    Returns:
      rho: float if p==1 else np.ndarray shape (p,)
      sigma: float
    """
    x = torch.as_tensor(x)
    u = torch.as_tensor(u)
    z = infer_z(flow, x, u, delta_t=delta_t, si=si, dt_full=dt_full, F=F)

    # ignore early steps (condition padding + AR lag warmup)
    stride = full_step_stride(si=si, dt_full=dt_full)
    burn_in = max(int(delta_t * stride), int(p))

    rho, sigma_i = fit_flow_latent_ar(
        z.detach().cpu().numpy(),
        p=p,
        method=method,
        enforce_stability=enforce_stability,
        burn_in=burn_in,
    )

    return rho, sigma_i
