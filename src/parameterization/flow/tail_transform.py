"""Implement flexible tail transforms for normalizing flows."""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TailTransformTTF(nn.Module):
    """
    Tail Transform Flow (TTF) final layer from Hickling & Prangle (2025).

    R(z; lambda+, lambda-) = mu + sigma * s/lambda_s * [ erfc(|z|/sqrt(2))^(-lambda_s) - 1 ]
    with s = sign(z), lambda_s = lambda+ if s=+1 else lambda-.

    We learn (mu, sigma, lambda+, lambda-) per dimension. lambda+, lambda- and sigma are constrained > 0.
    See paper eq. (3) and Appendix B.3.
    """

    def __init__(
        self,
        dim: int,
        init_mu: float = 0.0,
        init_sigma: float = 1.0,
        init_lambda: float = 0.1,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.dim = dim
        self.eps = float(eps)

        self.mu = nn.Parameter(torch.full((dim,), float(init_mu)))
        self.log_sigma = nn.Parameter(torch.log(torch.full((dim,), float(init_sigma))))

        # store raw params, map via softplus to enforce positivity
        self.raw_lambda_plus = nn.Parameter(self._inv_softplus(init_lambda))
        self.raw_lambda_minus = nn.Parameter(self._inv_softplus(init_lambda))

    def _inv_softplus(self, x: float) -> torch.Tensor:
        x_t = torch.tensor(float(x))
        return torch.log(torch.expm1(x_t))

    def _sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma).clamp_min(self.eps)

    def _lambda_plus(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda_plus).clamp_min(self.eps)

    def _lambda_minus(self) -> torch.Tensor:
        return F.softplus(self.raw_lambda_minus).clamp_min(self.eps)

    @staticmethod
    def _sign(x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward: z -> y, returns (y, log_det) with log_det shape (batch,).
        """
        mu = self.mu
        sigma = self._sigma()
        lam_p = self._lambda_plus()
        lam_m = self._lambda_minus()

        s = self._sign(z)  # (B,D)
        lam = torch.where(s > 0, lam_p, lam_m)  # (B,D) via broadcast

        a = torch.special.erfc(torch.abs(z) / math.sqrt(2.0)).clamp_min(self.eps)
        y = mu + sigma * (s / lam) * (a.pow(-lam) - 1.0)

        # log|dR/dz| (Appendix B.3): sigma * sqrt(2/pi) * exp(-z^2/2) * a^(-lambda-1)
        log_d = (
            torch.log(sigma)
            + 0.5 * math.log(2.0 / math.pi)
            - 0.5 * z.pow(2)
            + (-(lam + 1.0)) * torch.log(a)
        )
        return y, log_d.sum(dim=-1)

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse: y -> z, returns (z, log_det_inv) with log_det_inv shape (batch,).
        """
        mu = self.mu
        sigma = self._sigma()
        lam_p = self._lambda_plus()
        lam_m = self._lambda_minus()

        dy = y - mu
        s = self._sign(dy)
        lam = torch.where(s > 0, lam_p, lam_m)

        t = (lam * (torch.abs(dy) / sigma) + 1.0).clamp_min(self.eps)
        p = t.pow(-1.0 / lam).clamp(self.eps, 1.0 - self.eps)  # p = erfc(|z|/sqrt(2))

        # erfc^{-1}(p) using erfc(x)=2*Phi(-x*sqrt(2)) => erfc^{-1}(p) = -ndtri(p/2)/sqrt(2)
        w = (p * 0.5).clamp(self.eps, 1.0 - self.eps)
        erfc_inv = -torch.special.ndtri(w) / math.sqrt(2.0)

        z = s * math.sqrt(2.0) * erfc_inv

        # log|dR^{-1}/dy| = - log|dR/dz| evaluated at z
        _, log_det_fwd = self.forward(z)
        return z, -log_det_fwd
