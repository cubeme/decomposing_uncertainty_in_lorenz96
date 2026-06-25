"""Define conditional RealNVP flow models."""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from parameterization.flow.base_distribution import ARpBase, BaseDist, StdNormal
from parameterization.flow.tail_transform import TailTransformTTF


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Sequence[int] = (128, 128),
        activation=nn.ReLU,
    ):
        super().__init__()
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(activation())
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Conditional Affine Coupling Layer (RealNVP style, Dinh et al. 2017)
class ConditionalAffineCoupling(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        mask: torch.Tensor,
        hidden_dims: Sequence[int] = (128, 128),
    ):
        super().__init__()
        assert mask.shape[-1] == dim
        self.register_buffer("mask", mask)  # shape (dim,)
        # Input to MLP is [masked_x, cond], output is [log_s, t]
        self.net = MLP(in_dim=dim + cond_dim, out_dim=2 * dim, hidden_dims=hidden_dims)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        inverse: bool = False,
    ):
        """
        x: (batch, dim)
        cond: (batch, cond_dim)
        Returns:
          y: transformed tensor, same shape as x
          log_det: (batch,)
        """
        mask = self.mask  # (dim,)
        x_masked = x * mask  # part that stays fixed

        h = torch.cat([x_masked, cond], dim=-1)
        # Compute scaling s and translation t
        log_s, t = self.net(h).chunk(2, dim=-1)

        # Stabilize scaling
        log_s = torch.tanh(log_s)

        # Only apply to unmasked dimensions
        log_s_unmasked = log_s * (1.0 - mask)
        t_unmasked = t * (1.0 - mask)

        if not inverse:
            # Forward: base -> data
            y = x_masked + (1.0 - mask) * (x * torch.exp(log_s_unmasked) + t_unmasked)
            # Determinant of Jacobian is the sum of the diagonal elements, which corresponds to the elements of the scaling factor
            log_det = log_s_unmasked.sum(dim=-1)
        else:
            # Inverse: data -> base
            y = x_masked + (1.0 - mask) * (
                (x - t_unmasked) * torch.exp(-log_s_unmasked)
            )
            log_det = -log_s_unmasked.sum(dim=-1)

        return y, log_det


class ConditionalRealNVP(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        n_coupling_layers: int = 6,
        hidden_dims: Sequence[int] = (128, 128),
        use_flexible_tails: bool = False,
        ttf_init_lambda: float = 0.1,
        base_dist: Optional[BaseDist] = None,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.n_coupling_layers = n_coupling_layers
        self.hidden_dims = tuple(hidden_dims)
        self.use_flexible_tails = bool(use_flexible_tails)

        self.base = base_dist if base_dist is not None else StdNormal(dim)

        masks = []
        for i in range(n_coupling_layers):
            # Alternate masks 1,0,1,0,...
            if i % 2 == 0:
                mask = torch.cat([torch.ones(dim // 2), torch.zeros(dim - dim // 2)])
            else:
                mask = torch.cat([torch.zeros(dim // 2), torch.ones(dim - dim // 2)])
            masks.append(mask)

        self.coupling_layers = nn.ModuleList(
            [
                ConditionalAffineCoupling(
                    dim=dim,
                    cond_dim=cond_dim,
                    mask=m,
                    hidden_dims=hidden_dims,
                )
                for m in masks
            ]
        )

        self.tail = (
            TailTransformTTF(dim=dim, init_lambda=ttf_init_lambda)
            if self.use_flexible_tails
            else None
        )

    def f(self, z: torch.Tensor, cond: torch.Tensor):
        """Map base z -> data u. Note: In Dinh et al. notation, this is f^-1."""
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.coupling_layers:
            x, log_det = layer(x, cond, inverse=False)
            log_det_total += log_det

        if self.tail is not None:
            x, log_det_tail = self.tail.forward(x)
            log_det_total += log_det_tail

        return x, log_det_total

    def f_inv(self, u: torch.Tensor, cond: torch.Tensor):
        """Map data u -> base z. Note: In Dinh et al. notation, this is f."""
        log_det_total = torch.zeros(u.shape[0], device=u.device)
        x = u

        if self.tail is not None:
            x, log_det_tail_inv = self.tail.inverse(x)
            log_det_total += log_det_tail_inv

        for layer in reversed(self.coupling_layers):
            x, log_det = layer(x, cond, inverse=True)
            log_det_total += log_det

        return x, log_det_total

    def _apply_f_inv_flat(self, u: torch.Tensor, cond: torch.Tensor):
        """
        Apply f_inv to a sequence by flattening the time dimension.

        Expects:
            u    : (B, T, dim)
            cond : (B, T, cond_dim)

        Flattens to (B*T, dim), applies the per-step inverse flow,
        then reshapes back to:

            z       : (B, T, dim)
            log_det : (B, T)

        Returns
        -------
        z : torch.Tensor
            Latent variables corresponding to u.
        log_det : torch.Tensor
            Log-determinant per time step.
        """
        # u: (B,T,dim), cond: (B,T,cond_dim)
        B, T, D = u.shape
        u_flat = u.reshape(B * T, D)
        cond_flat = cond.reshape(B * T, cond.shape[-1])
        z_flat, log_det_flat = self.f_inv(u_flat, cond_flat)
        z = z_flat.reshape(B, T, D)
        log_det = log_det_flat.reshape(B, T)
        return z, log_det

    def log_prob_seq(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Log p(u | cond).
        u: (B,T,dim), cond: (B,T,cond_dim)
        returns: (B,)
        """
        z, log_det = self._apply_f_inv_flat(u, cond)
        log_pz = self.base.log_prob(z)
        # If StdNormal returns (B,T), sum over time; if ARpBase returns (B,), keep it.
        if log_pz.dim() == 2:
            log_pz = log_pz.sum(dim=-1)
        return log_pz + log_det.sum(dim=-1)

    def sample_seq(
        self, cond: torch.Tensor, z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Sample u ~ p(u | cond).
        cond: (B,T,cond_dim)
        z: optional (B,T,dim) for AR base, or (B,T,dim) even for Gaussian
        """
        B, T, _ = cond.shape
        device, dtype = cond.device, cond.dtype
        if z is None:
            z = self.base.sample((B, T, self.dim), device=device, dtype=dtype)
        # flatten time
        z_flat = z.reshape(B * T, self.dim)
        cond_flat = cond.reshape(B * T, cond.shape[-1])
        u_flat, _ = self.f(z_flat, cond_flat)
        return u_flat.reshape(B, T, self.dim)

    def get_config(self) -> dict:
        cfg = {
            "dim": self.dim,
            "cond_dim": self.cond_dim,
            "n_coupling_layers": self.n_coupling_layers,
            "hidden_dims": list(self.hidden_dims),
            "use_flexible_tails": self.use_flexible_tails,
            "ttf_init_lambda": float(getattr(self.tail, "init_lambda", 0.1))
            if self.tail is not None
            else 0.1,
            "base_dist_name": "std_normal",
        }
        # base dist config
        if isinstance(self.base, ARpBase):
            cfg["base_dist_name"] = "ar_p"
            cfg["ar_order"] = int(self.base.p)
            cfg["init_rho"] = [float(x) for x in self.base.rho.detach().cpu().tolist()]
            cfg["init_sigma"] = float(self.base.sigma.detach().cpu().item())

        return cfg
