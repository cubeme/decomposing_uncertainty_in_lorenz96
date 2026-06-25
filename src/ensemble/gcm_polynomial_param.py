"""Run reduced-model ensembles with polynomial parameterizations."""

import multiprocessing as mp

import numpy as np
from absl import logging

from ensemble.ensemble_utils import check_time_consistency, validate_init_states_shape
from models.GCM.gcm import GCM
from models.helpers import parse_time_stepping_func
from parameterization.baselines.polynomial_parameterization import (
    PolynomialParameterization,
)


def run_poly_param_member(args):
    """
    Run a single ensemble member using a manually stepped GCM with deterministic
    polynomial parameterization.

    Args:
        i_init (int): Index of the initial state.
        i_member (int): Index of the ensemble member.
        init_state (np.ndarray): Initial conditions for the state variables.
        f_schedule (ForcingSchedule): Forcing schedule for the GCM.
        si (float): Sampling interval.
        total_time (float): Total simulation time.
        coefs (np.ndarray): Polynomial coefficients for the parameterization.
        dt (float): Time step for numerical integration.
        log_components (bool): If True, log and return component histories.

    Returns:
        If log_components is True:
            tuple:
                - i_init (int): Index of the initial state.
                - i_member (int): Index of the ensemble member.
                - i_model (int): Index of the model.
                - x_pred (np.ndarray): Predicted state variables over time.
                - t (np.ndarray): Corresponding time points.
                - param_hist (np.ndarray): Parameterization history over time.
                - advection_hist (np.ndarray): Advection term history over time.
                - damping_hist (np.ndarray): Damping term history over time.
        Else:
            tuple:
                - i_init (int): Index of the initial state.
                - i_member (int): Index of the ensemble member.
                - i_model (int): Index of the model.
                - x_pred (np.ndarray): Predicted state variables over time.
                - t (np.ndarray): Corresponding time points.
    """
    (
        i_init,
        i_member,
        i_model,
        init_state,
        f_schedule,
        si,
        total_time,
        coefs,
        dt,
        time_stepping,
    ) = args

    gcm_model = GCM(PolynomialParameterization(coefs), F_schedule=f_schedule)
    time_stepping_func = parse_time_stepping_func(time_stepping)

    x_pred, t = gcm_model(
        init_state,
        si=si,
        total_time=total_time,
        dt=dt,
        time_stepping_func=time_stepping_func,
    )
    return i_init, i_member, i_model, x_pred, t


def run_poly_param_ensemble_parallel_multiprocessing(
    init_states,
    config,
    coefs,
    num_processes=mp.cpu_count(),
):
    """
    Run a polynomial-parameterized GCM ensemble in parallel with multiprocessing.

    Args:
        init_states (np.ndarray): Initial conditions of shape (N, M, L, K),
            where N = number of initial states, M = ensemble members,
            L = models, and K = state dimension.
        config (object): Model configuration containing time stepping,
            forcing schedule, dt, si, and total_time.
        coefs (np.ndarray): Polynomial coefficients for the deterministic
            component of the parameterization.
        num_processes (int, optional): Number of worker processes.

    Returns:
        tuple:
            - x_ens_forecast (np.ndarray): Forecast array of shape (N, M, L, time, K).
            - t (np.ndarray): Shared time array.
    """
    model_type = "deterministic" if coefs.ndim == 1 else "bayesian"

    N, M, L, K = validate_init_states_shape(config, init_states)

    if model_type == "bayesian":
        if coefs.shape[0:2] != (M, L) or coefs.ndim != 3:
            raise ValueError(
                f"Bayesian ensemble with n_ens_members={M} and ar_order={L} requires coefs shape (M, P, poly_order+1), got {coefs.shape}."
            )

    logging.info(
        "Run %s GCM ensemble (%d x %d x %d) for total_time=%s with %s method...",
        model_type,
        N,
        M,
        L,
        str(config.total_time),
        config.time_stepping,
    )

    nt = int(config.total_time / config.si)
    x_ens_forecast = np.zeros((N, M, L, nt + 1, K), dtype=np.float32)

    tasks = [
        (
            i_n,
            i_m,
            i_l,
            init_states[i_n, i_m, i_l],
            config.f_schedule,
            config.si,
            config.total_time,
            coefs if model_type == "deterministic" else coefs[i_m, i_l],
            config.dt,
            config.time_stepping,
        )
        for i_n in range(N)
        for i_m in range(M)
        for i_l in range(L)
    ]

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_poly_param_member, tasks)

    t0 = None
    for i_n, i_m, i_l, x_pred, t in results:
        x_ens_forecast[i_n, i_m, i_l] = x_pred
        if t0 is None:
            t0 = t
        else:
            check_time_consistency(t0, t)
    return x_ens_forecast, np.asarray(t0)
