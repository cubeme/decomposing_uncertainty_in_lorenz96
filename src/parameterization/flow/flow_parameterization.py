"""Apply trained normalizing flows as model parameterizations."""

from typing import Optional, Sequence

import torch
import torch.nn as nn

from parameterization.base_parameterization import BaseParameterization
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.training.data import full_step_stride


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.dim() == 1 else x


class FlowParameterization(BaseParameterization, nn.Module):
    def __init__(
        self,
        x_dim: int,
        u_dim: int,
        flow_model: ConditionalRealNVP,
        ar_order: int,
        rho: float | Sequence[float] = 0.0,
        sigma: Optional[float] = None,  # innovation std for AR(p) needed if p>0
        delta_t: int = 0,
        si: float = 1.0,
        dt_full: float = 1.0,
        include_forcing_in_cond: bool = False,
        device: Optional[torch.device] = None,
        seed: int = 0,
    ):
        """
        Latent dynamics:
          AR(0)/iid:  z_t ~ N(0, I) independently each predict() call
          AR(p):      z_{t+1} = sum_{i=1..p} rho_i z_{t+1-i} + sigma * eps

        Where sigma is chosen so each component has stationary variance ~1
        under the simplifying assumption the AR is stable:
          sigma = sqrt(max(0, 1 - sum rho_i^2))

        Notes:
          - AR(1) is recovered by rho=float and ar_order=1 (default behavior).
          - For AR(p), pass rho as a length-p sequence (list/tuple/tensor).
          - Call update() to advance latent state; predict() does not advance it.
        """
        nn.Module.__init__(self)
        BaseParameterization.__init__(self)

        self.device = device or torch.device("cpu")
        self.flow = flow_model.to(self.device)

        self.u_dim = int(u_dim)
        self.x_dim = int(x_dim)

        # parse rho
        if isinstance(rho, (float, int)):
            rho_vec = torch.tensor([float(rho)], dtype=torch.float32)
        else:
            rho_vec = torch.tensor([float(r) for r in rho], dtype=torch.float32)

        # check ar_order
        p = int(ar_order)

        if p < 0:
            raise ValueError(f"ar_order must be >= 0. Got {p}.")

        if p == 0:
            # AR(0)/iid: rho is unused; allow any rho input without forcing length=0
            rho_vec = torch.empty((0,), dtype=torch.float32)
        else:
            if rho_vec.numel() != p:
                raise ValueError(f"rho has length {rho_vec.numel()} but ar_order={p}.")

        self.ar_order = p  # p=0 means iid (AR(0))
        self.rho = rho_vec.to(self.device)  # (p,)

        # check innovation std; always required for AR(p) with p>0
        if p > 0 and sigma is None:
            raise ValueError("sigma must be provided for AR(p) with p>0.")
        self.sigma = None if sigma is None else float(sigma)

        # latent history buffer: list of tensors, newest first: [z_t, z_{t-1}, ...]
        self._z_hist = []

        self.delta_t = int(delta_t)  # number of full history steps
        self.stride = full_step_stride(si=si, dt_full=dt_full)
        self.include_forcing_in_cond = bool(include_forcing_in_cond)
        self._x_hist = []

        # noise generator
        self.noise_seed = int(seed)
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.noise_seed)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)

        # robustly infer current device after moving module
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            # flow might have no parameters in rare cases; fall back to flow device
            self.device = next(self.flow.parameters()).device

        self.flow = self.flow.to(self.device)
        self.rho = self.rho.to(self.device)

        # move latent history
        if len(self._z_hist) > 0:
            self._z_hist = [z.to(self.device) for z in self._z_hist]

        # recreate generator on new device
        self.noise_gen = torch.Generator(device=self.device)
        self.noise_gen.manual_seed(self.noise_seed)

        return self

    def reset(self):
        self._x_hist = []
        self._z_hist = []

    def _push_x(self, x: torch.Tensor) -> None:
        """Append current x to history and keep last (delta_t + 1) entries."""
        self._x_hist.append(x)
        max_len = self.delta_t * self.stride + 1
        if len(self._x_hist) > max_len:
            self._x_hist = self._x_hist[-max_len:]

    def _build_cond(self, F: Optional[float], dtype: torch.dtype) -> torch.Tensor:
        """
        Build conditioning vector for current time step.

        cond_t = [x_t, x_{t-1}, ..., x_{t-delta_t}, (optional) F_t]
        """
        if self.delta_t == 0:
            cond = self._x_hist[-1]
        else:
            blocks = []
            for j in range(self.delta_t + 1):
                idx = -1 - j * self.stride
                blocks.append(
                    self._x_hist[idx] if -idx <= len(self._x_hist) else self._x_hist[0]
                )
            cond = torch.cat(blocks, dim=-1)

        if self.include_forcing_in_cond:
            if self.include_forcing_in_cond and F is None:
                raise ValueError(
                    "F must be provided when include_forcing_in_cond=True."
                )
            F_tensor = torch.tensor(
                [[float(F)]], device=self.device, dtype=dtype
            ).expand(cond.shape[0], 1)
            cond = torch.cat([cond, F_tensor], dim=-1)

        return cond

    def _init_z_hist(self, batch_size: int, dtype: torch.dtype) -> None:
        """Initialize z history to length p with iid N(0,I). Newest first."""
        self._z_hist = []
        for _ in range(self.ar_order):
            z = torch.randn(
                (batch_size, self.u_dim),
                device=self.device,
                generator=self.noise_gen,
                dtype=dtype,
            )
            self._z_hist.append(z)

    def update(self):
        """
        Advance latent state z_t -> z_{t+1} using AR(p).

        If ar_order==0: nothing to update (iid case).
        If history not initialized yet: do nothing (will init in predict).
        """
        if self.ar_order <= 0:
            return
        if len(self._z_hist) == 0:
            return

        # z_{t+1} = sum_i rho_i z_{t+1-i} + sigma eps
        eps = torch.randn(
            self._z_hist[0].shape,
            device=self.device,
            generator=self.noise_gen,
            dtype=self._z_hist[0].dtype,
        )

        # _z_hist is [z_t, z_{t-1}, ..., z_{t-p+1}]
        z_next = 0.0
        for i in range(self.ar_order):
            z_next = z_next + self.rho[i] * self._z_hist[i]

        z_next = z_next + self.sigma * eps

        # shift history: new newest first
        self._z_hist = [z_next] + self._z_hist[: self.ar_order - 1]

    def predict(self, x: torch.Tensor, F: Optional[float] = None) -> torch.Tensor:
        """
        Given x_t, produce a sample u_t.

        - If ar_order==0: draws fresh iid z each call.
        - If ar_order>0: uses current z_t from history; does not advance it.
          Call update() to move to next latent time.
        """
        x = _ensure_batch(x).to(self.device)
        batch_size = x.shape[0]

        self._push_x(x)
        cond = self._build_cond(F, dtype=x.dtype)

        if self.ar_order > 0:
            if len(self._z_hist) == 0 or self._z_hist[0].shape[0] != batch_size:
                self._init_z_hist(batch_size=batch_size, dtype=cond.dtype)
            z = self._z_hist[0]  # z_t
        else:
            # iid latent noise
            z = torch.randn(
                (batch_size, self.u_dim),
                device=self.device,
                generator=self.noise_gen,
                dtype=cond.dtype,
            )

        cond_seq = cond.unsqueeze(1)  # (B, 1, cond_dim)
        z_seq = z.unsqueeze(1)  # (B, 1, u_dim)
        u = self.flow.sample_seq(cond=cond_seq, z=z_seq)  # (B, 1, u_dim)
        u = u[:, 0, :]  # back to (B, u_dim)

        return u.squeeze(0) if u.shape[0] == 1 else u

    def log_prob(
        self, x: torch.Tensor, u: torch.Tensor, F: Optional[float] = None
    ) -> torch.Tensor:
        """
        Evaluate log p_theta(U | X) for diagnostics / comparison.
        """
        x = _ensure_batch(x).to(self.device)
        u = _ensure_batch(u).to(self.device)

        self._push_x(x)
        cond = self._build_cond(F, dtype=x.dtype)

        cond_seq = cond.unsqueeze(1)  # (B, 1, cond_dim)
        u_seq = u.unsqueeze(1)  # (B, 1, u_dim)
        return self.flow.log_prob_seq(u_seq, cond_seq)  # (B,)
