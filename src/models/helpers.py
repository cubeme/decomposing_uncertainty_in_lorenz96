"""Provide shared model equations and time-stepping helpers."""

import numpy as np
from numba import jit

from models.GCM.time_stepping import RK2, RK4, euler_forward


@jit
def L96_2t_x_dot_y_dot(x, y, F, h, b, c):
    """
    Compute the time derivatives for the two-time-scale Lorenz '96 model manually.

    Equations:
        d/dt X[k] = -X[k-1] (X[k-2] - X[k+1]) - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] (Y[j+2] - Y[j-1]) - c Y[j] + h.c/b X[k]

    Args:
        x (np.ndarray): Slow variables (X). Shape: (K,).
        y (np.ndarray): Fast variables (Y). Shape: (K*J,).
        F (float): Forcing term.
        h (float): Coupling coefficient.
        b (float): Ratio of amplitudes.
        c (float): Time-scale ratio.

    Returns:
        tuple: A tuple containing:
            - x_dot (np.ndarray): Time derivatives of X variables. Shape: (K,).
            - y_dot (np.ndarray): Time derivatives of Y variables. Shape: (K*J,).
            - coupling (np.ndarray): Coupling term. Shape: (K,).
    """
    # Compute K and J from the data
    K = x.size
    J = y.size // K

    x_dot = np.zeros(x.shape)
    y_dot = np.zeros(y.shape)

    # Compute element-wise as this is faster with numba jit
    for k in range(K):
        x_dot[k] = (
            -x[k - 1] * (x[k - 2] - x[(k + 1) % K])
            - x[k]
            + F
            - h * c / b * np.sum(y[k * J : (k + 1) * J])
        )
    for j in range(J * K):
        y_dot[j] = (
            -c * b * y[(j + 1) % (J * K)] * (y[(j + 2) % (J * K)] - y[j - 1])
            - c * y[j]
            + h * c / b * x[int(j / J)]
        )

    return x_dot, y_dot


@jit
def L96_eq1_x_dot(x: np.ndarray, f: float) -> np.ndarray:
    """
    Calculate the time rate of change for the X variables in the
    single time scale Lorenz '96 model.

    Equation:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        x (np.ndarray): Values of X variables at the current time step. Shape:
            (K,).
        f (float): Forcing term F.
        advect (bool): Whether to include the advection term in the
            computation. Defaults to True.

    Returns:
        np.ndarray: Array of tendencies for the X variables. Shape: (K,).
    """
    K = x.size
    x_dot = np.zeros(K)

    for k in range(K):
        x_dot[k] = -x[k - 1] * (x[k - 2] - x[(k + 1) % K]) - x[k] + f

    return x_dot


def parse_time_stepping_func(time_stepping_func_str: str):
    """
    Parse a time stepping function string and return the corresponding function.

    Args:
        time_stepping_func_str (str): String identifier for the time stepping
            function. Supported values are:
            - "euler_forward": Forward Euler method
            - "RK2": Second-order Runge-Kutta method
            - "RK4": Fourth-order Runge-Kutta method

    Returns:
        callable: The corresponding time stepping function.

    Raises:
        ValueError: If the provided string does not match any known time
            stepping function.
    """
    if time_stepping_func_str == "euler_forward":
        return euler_forward
    elif time_stepping_func_str == "RK2":
        return RK2
    elif time_stepping_func_str == "RK4":
        return RK4
    else:
        raise ValueError(f"Unknown time stepping function {time_stepping_func_str}.")
