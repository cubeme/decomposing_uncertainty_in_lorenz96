import math

import numpy as np
import torch
import torch.nn as nn
from pytest import mark, raises

from parameterization.flow.fit_rho import fit_rho_sigma_p_from_data, infer_z
from parameterization.flow.training.data import build_condition


class _DummyFlow(nn.Module):
    """
    Minimal stand-in for ConditionalRealNVP.

    Inverse is defined as:
        z = f_inv(u | cond) = u - cond_base
    where cond_base is the first K components of cond.
    So if we construct u = cond_base + z, inferred latent is exactly z.
    """

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))  # provides device via parameters()
        self.last_cond = None

    def f_inv(self, u: torch.Tensor, cond: torch.Tensor):
        self.last_cond = cond.detach().cpu()
        cond_base = cond[:, : u.shape[1]]
        z = u - cond_base
        log_det = torch.zeros(u.shape[0], device=u.device)
        return z, log_det


def _make_arp_z(rho: list[float], steps: int, dim: int, sigma: float) -> torch.Tensor:
    """
    Generate AR(p):
        z_t = sum_{i=1..p} rho_i z_{t-i} + sigma * eps_t, eps ~ N(0, I)
    """
    p = len(rho)
    z = torch.zeros(steps, dim)
    z[:p] = torch.randn(p, dim)

    for t in range(p, steps):
        pred = 0.0
        for i in range(1, p + 1):
            pred = pred + rho[i - 1] * z[t - i]
        z[t] = pred + sigma * torch.randn(dim)

    return z


def test_infer_z_reconstructs_latent_with_history_and_forcing():
    torch.manual_seed(0)
    steps, dim = 500, 3
    delta_t = 2
    si = 0.5
    dt_full = 1.0

    z_true = _make_arp_z([0.45], steps=steps, dim=dim, sigma=0.3)
    x = torch.randn_like(z_true)
    F = torch.linspace(10.0, 20.0, steps)

    cond = build_condition(x, delta_t=delta_t, si=si, dt_full=dt_full, F=F)
    u = cond[:, :dim] + z_true

    flow = _DummyFlow()
    z_hat = infer_z(flow, x, u, delta_t=delta_t, si=si, dt_full=dt_full, F=F)

    torch.testing.assert_close(z_hat.cpu(), z_true, atol=1e-6, rtol=0)
    assert flow.last_cond is not None
    torch.testing.assert_close(flow.last_cond, cond.cpu())


def test_infer_z_raises_on_mismatched_lengths():
    flow = _DummyFlow()
    x = torch.randn(10, 3)
    u = torch.randn(9, 3)
    with raises(ValueError, match="same length T"):
        infer_z(flow, x, u)


@mark.parametrize("method", ["least_squares", "yule_walker"])
def test_fit_rho_sigma_p_from_data_recovers_p1(method):
    torch.manual_seed(1)
    rho = [0.6]
    sigma = math.sqrt(1.0 - rho[0] * rho[0])
    steps, dim = 2400, 4

    z_true = _make_arp_z(rho, steps=steps, dim=dim, sigma=sigma)
    x = torch.randn_like(z_true)
    cond = build_condition(x, delta_t=0, si=1.0, dt_full=1.0, F=None)
    u = cond[:, :dim] + z_true

    flow = _DummyFlow()
    rho_hat, sigma_hat = fit_rho_sigma_p_from_data(
        flow,
        x,
        u,
        p=1,
        method=method,
    )

    assert isinstance(rho_hat, float)
    assert abs(rho_hat - rho[0]) < 0.05
    assert abs(sigma_hat - sigma) < 0.08


@mark.parametrize("method", ["least_squares", "yule_walker"])
def test_fit_rho_sigma_p_from_data_recovers_p2(method):
    torch.manual_seed(2)
    rho = [0.5, -0.2]
    sigma = 0.4
    steps, dim = 5000, 3

    z_true = _make_arp_z(rho, steps=steps, dim=dim, sigma=sigma)
    x = torch.randn_like(z_true)
    cond = build_condition(x, delta_t=0, si=1.0, dt_full=1.0, F=None)
    u = cond[:, :dim] + z_true

    flow = _DummyFlow()
    rho_hat, sigma_hat = fit_rho_sigma_p_from_data(
        flow,
        x,
        u,
        p=2,
        method=method,
    )

    assert isinstance(rho_hat, np.ndarray)
    assert rho_hat.shape == (2,)
    np.testing.assert_allclose(rho_hat, np.array(rho), atol=0.07)
    assert abs(sigma_hat - sigma) < 0.1
