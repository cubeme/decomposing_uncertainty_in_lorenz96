"""Run sequential reduced-model ensembles with flow parameterizations."""

from typing import Optional

import numpy as np
import torch
from absl import logging

from ensemble.ensemble_utils import (
    check_ar_order_consistency,
    check_time_consistency,
    validate_init_states_shape,
)
from models.GCM.gcm import GCM
from models.helpers import parse_time_stepping_func
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.flow_parameterization import FlowParameterization


class _NumpyFlowWrapper:
    def __init__(self, flow_param: FlowParameterization):
        self.flow_param = flow_param

    def update(self):
        self.flow_param.update()

    def predict(self, x: np.ndarray, F: Optional[float] = None) -> np.ndarray:
        x_tensor = torch.as_tensor(
            x, device=self.flow_param.device, dtype=torch.float32
        )
        u_tensor = self.flow_param.predict(x_tensor, F)
        return u_tensor.detach().cpu().numpy()


@torch.inference_mode()
def run_flow_param_ensemble_sequential_single_gpu(
    init_states: np.ndarray,  # (N, M, L, K)
    config,
    flow_model: ConditionalRealNVP,  # single shared model
    seeds: np.ndarray,  # shape (N, M, L) for per-member seeds
    rho: float | np.ndarray,  # shared rho (float for AR(1), array for AR(p))
    sigma: Optional[float] = None,  # shared innovation std; required if ar_order > 0
    device: str = "cuda",
):
    """
    Sequential (no multiprocessing) flow-parameterized GCM ensemble on a single GPU,
    using a single shared flow model and shared rho, but different per-member seeds.


    Args:
        init_states (np.ndarray): Initial conditions of shape (N, M, L, K),
            where N = number of initial states, M = ensemble members,
            L = models, and K = state dimension.
        config (object): Model configuration containing time stepping,
            forcing schedule, dt, si, and total_time.
        flow_model (ConditionalRealNVP): The shared flow model.
        seeds (np.ndarray): Random seeds of shape (N, M, L) for different parameterization realizations.
        rho (float | np.ndarray): AR(p) coefficients for the stochastic noise process.
        sigma (float): Innovation standard deviation of the AR process.
        device (str): Device to run the flow model on.

    Returns:
        tuple:
            - x_ens_forecast (np.ndarray): Forecast array of shape (N, M, L, time, K).
            - t (np.ndarray): Shared time array.
    """
    N, M, L, K = validate_init_states_shape(config, init_states)

    seeds = np.asarray(seeds)
    if seeds.shape != (N, M, L):
        raise ValueError(
            f"Expected seeds shape (N, M, L) with N={N}, M={M}, L={L}, got {seeds.shape}"
        )

    logging.info(
        "Run flow GCM ensemble (%d x %d x %d) for total_time=%s with %s method...",
        N,
        M,
        L,
        str(config.total_time),
        config.time_stepping,
    )

    dev = torch.device(device)
    fm = flow_model.to(dev).eval()

    nt = int(config.total_time / config.si)
    x_ens_forecast = np.zeros((N, M, L, nt + 1, K), dtype=np.float32)

    time_stepping_func = parse_time_stepping_func(config.time_stepping)

    delta_t = config.delta_t
    include_forcing_in_cond = config.include_forcing_in_cond

    x_base_dim = int(config.K)
    assert x_base_dim == K, (
        f"State dimension K={K} from init_states does not match config.K={config.K}"
    )
    cond_dim = x_base_dim * (delta_t + 1) + (1 if include_forcing_in_cond else 0)

    # ar_order must be specified by config; do not infer from rho
    ar_order = int(config.ar_order)
    rho_arr = np.asarray(rho)
    check_ar_order_consistency(ar_order, rho_arr, sigma)

    # prepare rho in the format expected by FlowParameterization
    if ar_order <= 1:
        # scalar rho for AR(0)/AR(1) (AR(0) ignores it)
        rho_ = float(rho_arr.reshape(-1)[0]) if rho_arr.size > 0 else 0.0
    else:
        # vector rho for AR(p>1)
        rho_ = rho_arr

    t0 = None
    for i_n in range(N):
        for i_m in range(M):
            for i_l in range(L):
                flow_param = FlowParameterization(
                    x_dim=cond_dim,
                    u_dim=fm.dim,
                    flow_model=fm,
                    ar_order=ar_order,
                    rho=rho_,
                    sigma=sigma,
                    delta_t=delta_t,
                    si=config.si,
                    dt_full=config.dt_full,
                    include_forcing_in_cond=include_forcing_in_cond,
                    device=dev,
                    seed=int(seeds[i_n, i_m, i_l]),
                )

                gcm_model = GCM(
                    _NumpyFlowWrapper(flow_param), F_schedule=config.f_schedule
                )

                x_pred, t = gcm_model(
                    init_states[i_n, i_m, i_l],
                    si=config.si,
                    total_time=config.total_time,
                    dt=config.dt,
                    time_stepping_func=time_stepping_func,
                )

                x_ens_forecast[i_n, i_m, i_l] = x_pred
                if t0 is None:
                    t0 = t
                else:
                    check_time_consistency(t0, t)

    return x_ens_forecast, np.asarray(t0)
