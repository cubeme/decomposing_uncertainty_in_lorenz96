"""Provide shared parameterization calculations."""

import numpy as np


def compute_coupling_from_x(x, dt, F, h, b, c):
    x_tmp = x[:-1]
    u_train = np.zeros_like(x_tmp)

    x_tmp_p1 = x[1:]
    dx_dt = (x_tmp_p1 - x_tmp) / dt

    F = F[:-1]
    # Expand dimension if needed
    if F.ndim == 1:
        F = F[:, None]

    u_train = (
        -np.roll(x_tmp, 1, axis=1)
        * (np.roll(x_tmp, 2, axis=1) - np.roll(x_tmp, -1, axis=1))
        - x_tmp
        + F
        - dx_dt
    )

    uscale = h * c / b
    u_train *= uscale

    return u_train, x_tmp


def compute_ar_p_noise(
    rho: float | np.ndarray,
    sigma: float,
    steps: int,
    k: int,
    seed: int = 17,
) -> np.ndarray:
    """
    Generate AR(p) noise for a specified number of time steps.

    Model:
        e_t = sum_{i=1..p} rho_i e_{t-i} + sigma * z_t,
    where z_t ~ N(0, I).

    Args:
        rho (float or np.ndarray): AR coefficients. Scalar for AR(1),
            or array-like of length p for AR(p).
        sigma (float): Innovation standard deviation.
        steps (int): Number of time steps to generate noise for.
        k (int): Number of variables.
        seed (int, optional): Random seed for reproducibility. Default is 17.

    Returns:
        numpy.ndarray: Generated AR(p) noise of shape (steps, k).
    """

    rho_arr = np.asarray(rho, dtype=float).reshape(-1)
    p = int(rho_arr.size)

    if p < 1:
        raise ValueError("rho must have length >= 1.")

    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("sigma must be a finite nonnegative float.")

    rg = np.random.default_rng(seed=seed)

    noise = np.zeros((steps, k), dtype=float)

    # Initialize first p steps with iid N(0, sigma^2) for simplicity
    init_steps = min(p, steps)
    if init_steps > 0:
        noise[:init_steps, :] = sigma * rg.normal(0, 1, size=(init_steps, k))

    # Recursion
    for t in range(p, steps):
        e_next = 0.0
        for i in range(p):
            e_next = e_next + rho_arr[i] * noise[t - i - 1]
        e_next = e_next + sigma * rg.normal(0, 1, size=k)
        noise[t, :] = e_next

    return noise
