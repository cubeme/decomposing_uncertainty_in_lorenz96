"""Define latent base distributions for normalizing flows."""

import math
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


class BaseDist(nn.Module):
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(self, shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        raise NotImplementedError


class StdNormal(BaseDist):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., dim)
        return -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)

    def sample(self, shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype)


class ARpBase(nn.Module):
    """
    Independent AR(p) per latent dimension.

    z_t = sum_{i=1..p} rho_i z_{t-i} + sigma * eps_t
    eps_t ~ N(0, I_dim)

    Stationarity enforced via reflection (PACF) coefficients kappa_i
    with |kappa_i| < 1.

    Convention: z is (B, T, dim).
    For t <= p, we treat z_{1:p} as iid N(0,1).
    """

    def __init__(
        self,
        dim: int,
        p: int,
        init_rho: Optional[Union[float, Sequence[float]]] = None,
        init_sigma: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.p = int(p)
        if self.p < 1:
            raise ValueError(f"p must be >= 1. Got p={self.p}.")

        # -------------------------
        # Normalize init_rho
        # -------------------------
        if init_rho is None:
            init_rho_list = [0.0] * self.p
        elif isinstance(init_rho, (int, float)):
            if self.p != 1:
                raise ValueError(f"Scalar init_rho only valid for p=1. Got p={self.p}.")
            init_rho_list = [float(init_rho)]
        else:
            init_rho_list = [float(r) for r in init_rho]
            if len(init_rho_list) != self.p:
                raise ValueError(
                    f"init_rho has length {len(init_rho_list)} but p={self.p}."
                )

        # -------------------------
        # Convert rho -> kappa (step-down recursion)
        # -------------------------
        init_rho_tensor = torch.tensor(init_rho_list, dtype=torch.float32)
        init_kappa = self._ar_to_pacf(init_rho_tensor)

        # Clamp for numerical stability before atanh
        eps = 1e-6
        init_kappa = init_kappa.clamp(-1 + eps, 1 - eps)

        self.raw_kappa = nn.Parameter(torch.atanh(init_kappa))  # (p,)

        # -------------------------
        # Sigma parameterization
        # -------------------------
        if init_sigma <= 0:
            raise ValueError(f"init_sigma must be > 0. Got {init_sigma}.")
        self.log_sigma = nn.Parameter(
            torch.tensor(math.log(float(init_sigma)), dtype=torch.float32)
        )

    # ============================================================
    # Properties
    # ============================================================

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    @property
    def kappa(self) -> torch.Tensor:
        # Strictly inside (-1,1)
        return 0.999 * torch.tanh(self.raw_kappa)

    @property
    def rho(self) -> torch.Tensor:
        return self._pacf_to_ar(self.kappa)

    # ============================================================
    # Levinson Recursions
    # ============================================================

    @staticmethod
    def _pacf_to_ar(kappa: torch.Tensor) -> torch.Tensor:
        """
        Step-up recursion:
        PACF -> AR coefficients.

        kappa: (p,)
        returns rho: (p,)
        """
        p = int(kappa.shape[0])
        a = kappa.new_zeros((0,))

        for m in range(1, p + 1):
            km = kappa[m - 1]
            if m == 1:
                a = km.view(1)
            else:
                a_rev = torch.flip(a, dims=[0])
                a_new = a - km * a_rev
                a = torch.cat([a_new, km.view(1)], dim=0)

        return a

    @staticmethod
    def _ar_to_pacf(rho: torch.Tensor) -> torch.Tensor:
        """
        Step-down recursion:
        AR coefficients -> PACF.

        rho: (p,)
        returns kappa: (p,)
        """
        p = int(rho.shape[0])
        a = rho.clone()
        kappa = rho.new_zeros(p)

        for m in reversed(range(1, p + 1)):
            km = a[m - 1]
            kappa[m - 1] = km

            if m == 1:
                break

            a_new = a[: m - 1].clone()
            denom = 1.0 - km * km
            if torch.abs(denom) < 1e-8:
                denom = torch.sign(denom) * 1e-8

            for i in range(m - 1):
                a_new[i] = (a[i] + km * a[m - 2 - i]) / denom

            a[: m - 1] = a_new

        return kappa

    # ============================================================
    # AR Process
    # ============================================================

    def _innovations(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        p = self.p
        if T <= p:
            return z.new_zeros((B, 0, D))

        preds = 0.0
        for i in range(1, p + 1):
            preds = preds + self.rho[i - 1] * z[:, p - i : T - i, :]
        return z[:, p:, :] - preds

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        B, T, D = z.shape
        p = self.p

        lp_init = -0.5 * (z[:, : min(p, T), :].pow(2) + math.log(2.0 * math.pi)).sum(
            dim=-1
        ).sum(dim=-1)

        if T <= p:
            return lp_init

        eps = self._innovations(z)
        sig = self.sigma

        lp_eps = -0.5 * (
            (eps / sig).pow(2) + math.log(2.0 * math.pi) + 2.0 * self.log_sigma
        ).sum(dim=-1)
        lp_eps = lp_eps.sum(dim=-1)

        return lp_init + lp_eps

    def sample(self, shape: Tuple[int, ...], device=None, dtype=None) -> torch.Tensor:
        B, T, D = shape
        assert D == self.dim
        p = self.p

        z = torch.zeros(B, T, D, device=device, dtype=dtype)
        t0 = min(p, T)

        if t0 > 0:
            z[:, :t0, :] = torch.randn(B, t0, D, device=device, dtype=dtype)

        if T <= p:
            return z

        sig = self.sigma.to(device=device, dtype=dtype)
        rho = self.rho.to(device=device, dtype=dtype)

        for t in range(p, T):
            pred = 0.0
            for i in range(1, p + 1):
                pred = pred + rho[i - 1] * z[:, t - i, :]
            z[:, t, :] = pred + sig * torch.randn(B, D, device=device, dtype=dtype)

        return z
