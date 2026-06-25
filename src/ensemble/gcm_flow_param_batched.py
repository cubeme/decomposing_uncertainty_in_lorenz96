"""Run batched reduced-model ensembles with flow parameterizations."""

from typing import Optional

import numpy as np
import torch
from absl import logging

from ensemble.ensemble_utils import (
    check_ar_order_consistency,
    validate_init_states_shape,
)
from ensemble.torch_utils import (
    RK2_torch,
    RK4_torch,
    euler_forward_torch,
    l96_rhs_torch,
)
from models.forcing_schedule import forcing_at
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.training.data import full_step_stride


class BatchedFlowParameterizationSharedFlow:
    """
    Batched flow parameterization with:
      - a single shared flow model
      - a single shared rho
      - per-(init, member, model) RNG stream (seeds shape N x M x L)
      - per-(init, member, model) latent AR(p) state history z_hist (each element shape B x u_dim)

    Interpretation:
      Each realization (i_n, i_m, i_l) has its own seed/latent state.
    """

    def __init__(
        self,
        flow_model: ConditionalRealNVP,
        ar_order: int,
        rho: float | np.ndarray,
        seeds: np.ndarray,  # shape (N, M, L)
        N: int,
        M: int,
        L: int,
        device: torch.device,
        x_base_dim: int,
        sigma: Optional[float] = None,
        delta_t: int = 0,
        si: float = 1.0,
        dt_full: float = 1.0,
        include_forcing_in_cond: bool = False,
    ):
        self.device = device
        self.N = int(N)
        self.M = int(M)
        self.L = int(L)
        self.B = self.N * self.M * self.L  # batch size

        seeds = np.asarray(seeds)
        if seeds.shape != (self.N, self.M, self.L):
            raise ValueError(
                f"seeds must have shape (N, M, L) with N={self.N}, M={self.M}, L={self.L}, got {seeds.shape}"
            )
        seeds_flat = seeds.reshape(self.B)

        self.flow = flow_model.to(device).eval()
        self.x_dim = int(self.flow.cond_dim)
        self.u_dim = int(self.flow.dim)

        self.x_base_dim = int(x_base_dim)
        self.delta_t = int(delta_t)  # number of full history steps
        self.stride = full_step_stride(si=si, dt_full=dt_full)
        self.include_forcing_in_cond = bool(include_forcing_in_cond)
        self._x_hist: list[torch.Tensor] = []

        self.ar_order = int(ar_order)
        rho_arr = np.asarray(rho)
        check_ar_order_consistency(self.ar_order, rho_arr, sigma)

        if self.ar_order <= 1:
            rho_val = float(rho_arr.reshape(-1)[0]) if rho_arr.size > 0 else 0.0
            self.rho = torch.tensor(
                rho_val, device=device, dtype=torch.float32
            )  # scalar
            self.rho_vec = None
        else:
            self.rho = None
            self.rho_vec = torch.as_tensor(
                rho_arr, device=device, dtype=torch.float32
            )  # (p,)

        # innovation std:
        # - required for AR(p>=1)
        # - unused for AR(0)
        self.sigma = None if sigma is None else float(sigma)

        # Per-realization RNG streams (one per batch element)
        self.gens: list[torch.Generator] = []
        for s in seeds_flat.tolist():
            g = torch.Generator(device=device)
            g.manual_seed(int(s))
            self.gens.append(g)

        # Per-realization latent AR(p) state history:
        # newest first: [z_t, z_{t-1}, ...], each (B, u_dim)
        self.z_hist: list[torch.Tensor] = []

    def _randn_per_realization(self, shape: tuple[int, int]) -> torch.Tensor:
        """
        Draw N(0,1) with independent per-realization generators.
        shape must be (B, D).
        """
        B, D = shape
        if B != self.B:
            raise ValueError(f"Expected B={self.B}, got {B}")
        zs = []
        for b in range(B):
            zs.append(torch.randn((D,), device=self.device, generator=self.gens[b]))
        return torch.stack(zs, dim=0)  # (B, D)

    def _push_x(self, x_flat: torch.Tensor) -> None:
        self._x_hist.append(x_flat)
        max_len = self.delta_t * self.stride + 1
        if len(self._x_hist) > max_len:
            self._x_hist = self._x_hist[-max_len:]

    def _build_cond(self, F: Optional[float], dtype: torch.dtype) -> torch.Tensor:
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
            if F is None:
                raise ValueError(
                    "F must be provided when include_forcing_in_cond=True."
                )
            F_tensor = torch.tensor(
                [[float(F)]], device=self.device, dtype=dtype
            ).expand(cond.shape[0], 1)
            cond = torch.cat([cond, F_tensor], dim=-1)
        return cond

    def _init_z_hist(self) -> None:
        """Initialize per-realization latent history to length p with iid N(0,I). Newest first."""
        self.z_hist = []
        for _ in range(self.ar_order):
            self.z_hist.append(
                self._randn_per_realization((self.B, self.u_dim))
            )  # (B,u_dim)

    def predict(self, x_flat: torch.Tensor, F: Optional[float] = None) -> torch.Tensor:
        """
        x_flat: (B, x_base_dim) where B = N*M*L
        returns u_flat: (B, u_dim)
        """
        if x_flat.shape != (self.B, self.x_base_dim):
            raise ValueError(
                f"Expected x_flat shape {(self.B, self.x_base_dim)}, got {tuple(x_flat.shape)}"
            )

        self._push_x(x_flat)
        x_cond = self._build_cond(F, dtype=x_flat.dtype)

        # Choose z per realization:
        # - if ar_order>0: reuse current z_t from history
        # - else: fresh z each call
        if self.ar_order > 0:
            if len(self.z_hist) == 0:
                self._init_z_hist()
            z = self.z_hist[0]  # (B,u_dim)
        else:
            z = self._randn_per_realization((self.B, self.u_dim))  # (B,u_dim)

        # Shared flow: do one flow call for the full batch
        cond_bt = x_cond.reshape(self.B, self.x_dim).unsqueeze(1)  # (B, 1, cond_dim)
        z_bt = z.reshape(self.B, self.u_dim).unsqueeze(1)  # (B, 1, u_dim)

        u_bt = self.flow.sample_seq(cond=cond_bt, z=z_bt)  # (B, 1, u_dim)
        u_flat = u_bt[:, 0, :]  # (B, u_dim)
        return u_flat

    def update(self):
        """
        Update per-realization latent AR state history.
        Only if ar_order>0.
        """
        if self.ar_order <= 0:
            return
        if len(self.z_hist) == 0:
            return

        eps = self._randn_per_realization((self.B, self.u_dim))  # (B,u_dim)

        if self.ar_order == 1:
            rho = self.rho  # scalar
            z_next = rho * self.z_hist[0] + self.sigma * eps
            self.z_hist = [z_next]
            return

        # AR(p>1): z_{t+1} = sum_i rho_i z_{t+1-i} + sigma eps
        rho_vec = self.rho_vec  # (p,)
        z_next = 0.0
        for i in range(self.ar_order):
            z_next = z_next + rho_vec[i] * self.z_hist[i]
        z_next = z_next + self.sigma * eps

        self.z_hist = [z_next] + self.z_hist[: self.ar_order - 1]


@torch.inference_mode()
def run_flow_param_ensemble_batched_single_gpu(
    init_states: np.ndarray,  # (N, M, L, K)
    config,
    flow_model: ConditionalRealNVP,  # single model
    seeds: np.ndarray,  # shape (N, M, L)
    rho: float | np.ndarray,  # shared rho (float for AR(1), array for AR(p))
    sigma: Optional[float] = None,  # shared innovation std; required if ar_order > 1
    device: str = "cuda",
    time_stepping: str = "RK2",  # "euler_forward" | "RK2" | "RK4"
):
    # validate_init_states_shape has already been adjusted
    N, M, L, K = validate_init_states_shape(config, init_states)

    seeds = np.asarray(seeds)
    if seeds.shape != (N, M, L):
        raise ValueError(
            f"Expected seeds shape (N, M, L) with N={N}, M={M}, L={L}, got {seeds.shape}"
        )

    logging.info(
        "Run flow-parameterized GCM ensemble (%d x %d x %d) for total_time=%s with %s method...",
        N,
        M,
        L,
        str(config.total_time),
        time_stepping,
    )

    B = N * M * L
    dev = torch.device(device)

    # Time grid and integration parameters
    nt = int(config.total_time / config.si)

    if config.si < config.dt:
        dt, ns = config.si, 1
    else:
        ns = int(config.si / config.dt + 0.5)
        assert abs(ns * config.dt - config.si) < 1e-14, (
            f"si is not an integer multiple of dt: si={config.si} dt={config.dt} ns={ns}"
        )
        dt = config.dt

    # Choose stepping function
    if time_stepping == "euler_forward":
        stepper = euler_forward_torch
    elif time_stepping == "RK2":
        stepper = RK2_torch
    elif time_stepping == "RK4":
        stepper = RK4_torch
    else:
        raise ValueError(f"Unsupported time_stepping: {time_stepping}")

    # Flatten init states to (B,K) on GPU
    X = torch.as_tensor(init_states.reshape(B, K), device=dev, dtype=torch.float32)

    # Parameterization: shared flow + shared rho, per-(N,M,L) RNG/z
    param = BatchedFlowParameterizationSharedFlow(
        flow_model=flow_model,
        ar_order=int(config.ar_order),
        rho=rho,
        sigma=sigma,
        seeds=seeds,  # (N, M, L)
        N=N,
        M=M,
        L=L,
        device=dev,
        x_base_dim=K,
        delta_t=config.delta_t,
        si=config.si,
        dt_full=config.dt_full,
        include_forcing_in_cond=config.include_forcing_in_cond,
    )

    # Output buffers
    X_hist = torch.empty((nt + 1, B, K), device=dev, dtype=X.dtype)
    T_hist = torch.empty((nt + 1,), device=dev, dtype=torch.float32)
    X_hist[0] = X
    T_hist[0] = 0.0

    def rhs(x: torch.Tensor, t: float) -> torch.Tensor:
        F_t = forcing_at(config.f_schedule, t)
        return l96_rhs_torch(x, F_t) - param.predict(x, F_t)

    for n in range(nt):
        t_n = config.si * n
        for s in range(ns):
            t_ns = t_n + s * dt
            X = stepper(rhs, dt, X, t_ns)

        param.update()

        X_hist[n + 1] = X
        T_hist[n + 1] = config.si * (n + 1)

    X_hist_np = X_hist.permute(1, 0, 2).detach().cpu().numpy()  # (B, nt+1, K)
    T_hist_np = T_hist.detach().cpu().numpy()  # (nt+1,)

    x_ens_forecast = X_hist_np.reshape(N, M, L, nt + 1, K)
    t = T_hist_np
    return x_ens_forecast, t
