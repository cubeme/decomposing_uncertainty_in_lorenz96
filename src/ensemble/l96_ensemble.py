"""Run fully resolved Lorenz '96 ensembles in parallel."""

import multiprocessing as mp
from multiprocessing import Pool

import numpy as np
from absl import logging

from models.L96.lorenz96 import L96


def run_single_l96(args):
    """
    Run a single L96 simulation for a given initial state using manual time stepping.

    Args:
        args (tuple): A tuple containing:
            - x_init (np.ndarray): Initial conditions for the slow variables (X).
            - y_init (np.ndarray): Initial conditions for the fast variables (Y).
            - t_init (float): Initial time.
            - k (int): Number of slow variables (X).
            - j (int): Number of fast variables (Y) per slow variable.
            - f_schedule (ForcingSchedule): Forcing schedule.
            - h (float): Coupling coefficient.
            - b (float): Spatial-scale ratio.
            - c (float): Time-scale ratio.
            - si (float): Sampling interval.
            - total_time (float): Total simulation time.
            - dt (float): Time step for numerical integration.

    Returns:
        tuple:
            - x (np.ndarray): Slow variables (X) over time.
            - y (np.ndarray): Fast variables (Y) over time.
            - t (np.ndarray): Time points.
            - u (np.ndarray): Coupling term history over time.
    """

    x_init, y_init, k, j, f_schedule, h, b, c, si, total_time, dt = args

    # Initialize the L96 model
    # Seed is irrelevant, as initial state gets set
    model = L96(k, j, h, b, c, F_schedule=f_schedule, t=0, seed=0)
    model.set_state(x_init, y_init)

    # Run the simulation
    x, y, t = model.run(si=si, total_time=total_time, dt=dt)

    return x, y, t


def run_l96_parallel(
    x_init_states,
    y_init_states,
    config,
    num_processes=mp.cpu_count(),
):
    """
    Run multiple L96 simulations in parallel using multiprocessing.

    Args:
        x_init_states (np.ndarray): Array of initial slow variable states, shape (n_init_states, n_ens_members, K).
        y_init_states (np.ndarray): Array of initial fast variable states, shape (n_init_states, n_ens_members, K*J).
        config (object): Configuration object with simulation parameters:
            - K (int): Number of slow variables (X).
            - J (int): Number of fast variables (Y) per slow variable.
            - f_schedule (ForcingSchedule): Forcing schedule.
            - h (float): Coupling coefficient.
            - b (float): Spatial-scale ratio.
            - c (float): Time-scale ratio.
            - si (float): Sampling interval.
            - total_time (float): Total simulation time..
            - dt (float): Time step for manual integration.
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
        log_components (bool, optional): If True, log component histories.
        save_path (str, optional): Path to save component logs.

    Returns:
        tuple: A tuple containing:
            - x_per_state (np.ndarray): Array of slow variables (X), shape (n_init_states, n_ens_members, time, K).
            - y_per_state (np.ndarray): Array of fast variables (Y), shape (n_init_states, n_ens_members, time, K*J).
            - t_per_state (np.ndarray): Array of time points, shape (time,).
    """
    if x_init_states.ndim != 3:
        raise ValueError(f"init_states must be 3D, got shape {x_init_states.shape}")

    N, M, _ = x_init_states.shape
    # Since this can be called in split mode, N and M are not necessarily
    # the total number of init states and ensemble members

    logging.info(
        "Run L96 ensemble with %d init states and %d ensemble members (%d total runs) for total_time=%s ...",
        N,
        M,
        N * M,
        str(config.total_time),
    )

    # Flatten (N, M, ...) -> (N*M, ...)
    x_flat = x_init_states.reshape(N * M, x_init_states.shape[2])
    y_flat = y_init_states.reshape(N * M, y_init_states.shape[2])
    del x_init_states, y_init_states

    tasks = [
        (
            x_flat[i],
            y_flat[i],
            config.K,
            config.J,
            config.f_schedule,
            config.h,
            config.b,
            config.c,
            config.si,
            config.total_time,
            config.dt,
        )
        for i in range(x_flat.shape[0])
    ]

    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_l96, tasks)

    # results: list of (x_i[T,K], y_i[T,JK], t_i[T])
    x_per_state, y_per_state, t_per_state = zip(*results)
    del x_flat, y_flat

    x_arr = np.asarray(x_per_state)  # (N*M, T, K)
    y_arr = np.asarray(y_per_state)  # (N*M, T, JK)
    t0 = np.asarray(t_per_state[0])  # (T,)
    del x_per_state, y_per_state

    # Sanity check that all t are identical
    # (cheap-ish check: compare shapes + first/last values)
    for ti in t_per_state[1:]:
        ti = np.asarray(ti)
        if ti.shape != t0.shape or ti[0] != t0[0] or ti[-1] != t0[-1]:
            raise ValueError(
                "Time arrays differ across ensemble members; cannot compress to one t."
            )

    # Reshape back to (N, M, T, ...)
    x_out = x_arr.reshape(N, M, *x_arr.shape[1:])
    y_out = y_arr.reshape(N, M, *y_arr.shape[1:])

    # Clean up
    del results, t_per_state, x_arr, y_arr

    return x_out, y_out, t0
