"""Implement Runge-Kutta integration for Lorenz '96 dynamics."""

import numpy as np
from numba import njit

from models.forcing_schedule import F_linear, F_osc
from models.helpers import (
    L96_2t_x_dot_y_dot,
)


@njit(cache=True)
def run_rk4_const(nt, dt, si, t_0, X, Y, h, c, b, F):
    # Result arrays
    x_hist, y_hist = (
        np.zeros((nt + 1, *X.shape)),
        np.zeros((nt + 1, *Y.shape)),
    )

    time = t_0 + np.zeros((nt + 1))

    X, Y = X.copy(), Y.copy()

    x_hist[0] = X
    y_hist[0] = Y

    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert abs(ns * dt - si) < 1e-14, (
            f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"
        )

    for n in range(nt):
        for s in range(ns):
            # RK4 update of X,Y
            x_dot1, y_dot1 = L96_2t_x_dot_y_dot(X, Y, F, h, b, c)
            x_dot2, y_dot2 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot1,
                Y + 0.5 * dt * y_dot1,
                F,
                h,
                b,
                c,
            )
            x_dot3, y_dot3 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot2,
                Y + 0.5 * dt * y_dot2,
                F,
                h,
                b,
                c,
            )
            x_dot4, y_dot4 = L96_2t_x_dot_y_dot(
                X + dt * x_dot3, Y + dt * y_dot3, F, h, b, c
            )

            X = X + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))
            Y = Y + (dt / 6.0) * ((y_dot1 + y_dot4) + 2.0 * (y_dot2 + y_dot3))

        x_hist[n + 1] = X
        y_hist[n + 1] = Y
        time[n + 1] = t_0 + si * (n + 1)

    return x_hist, y_hist, time


@njit(cache=True)
def run_rk4_linear(
    nt,
    dt,
    si,
    t_0,
    X,
    Y,
    h,
    c,
    b,
    F0,
    F1,
    t0,
    t1,
):
    # Result arrays
    x_hist, y_hist = (
        np.zeros((nt + 1, *X.shape)),
        np.zeros((nt + 1, *Y.shape)),
    )

    time = t_0 + np.zeros((nt + 1))

    X, Y = X.copy(), Y.copy()

    x_hist[0] = X
    y_hist[0] = Y

    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert abs(ns * dt - si) < 1e-14, (
            f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"
        )

    for n in range(nt):
        t_n = t_0 + si * n

        for s in range(ns):
            t_ns = t_n + s * dt

            # RK4 update of X,Y
            x_dot1, y_dot1 = L96_2t_x_dot_y_dot(
                X, Y, F_linear(t_ns, F0, F1, t0, t1), h, b, c
            )

            x_dot2, y_dot2 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot1,
                Y + 0.5 * dt * y_dot1,
                F_linear(t_ns + 0.5 * dt, F0, F1, t0, t1),
                h,
                b,
                c,
            )

            x_dot3, y_dot3 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot2,
                Y + 0.5 * dt * y_dot2,
                F_linear(t_ns + 0.5 * dt, F0, F1, t0, t1),
                h,
                b,
                c,
            )
            x_dot4, y_dot4 = L96_2t_x_dot_y_dot(
                X + dt * x_dot3,
                Y + dt * y_dot3,
                F_linear(t_ns + dt, F0, F1, t0, t1),
                h,
                b,
                c,
            )

            X = X + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))
            Y = Y + (dt / 6.0) * ((y_dot1 + y_dot4) + 2.0 * (y_dot2 + y_dot3))

        x_hist[n + 1] = X
        y_hist[n + 1] = Y
        time[n + 1] = t_0 + si * (n + 1)

    return x_hist, y_hist, time


@njit(cache=True)
def run_rk4_oscillating(nt, dt, si, t_0, X, Y, h, c, b, Fmean, amp, freq):
    # Result arrays
    x_hist, y_hist = (
        np.zeros((nt + 1, *X.shape)),
        np.zeros((nt + 1, *Y.shape)),
    )

    time = t_0 + np.zeros((nt + 1))

    X, Y = X.copy(), Y.copy()

    x_hist[0] = X
    y_hist[0] = Y

    if si < dt:
        dt, ns = si, 1
    else:
        ns = int(si / dt + 0.5)
        assert abs(ns * dt - si) < 1e-14, (
            f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"
        )

    for n in range(nt):
        t_n = t_0 + si * n

        for s in range(ns):
            t_ns = t_n + s * dt

            # RK4 update of X,Y
            x_dot1, y_dot1 = L96_2t_x_dot_y_dot(
                X, Y, F_osc(t_ns, Fmean, amp, freq), h, b, c
            )
            x_dot2, y_dot2 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot1,
                Y + 0.5 * dt * y_dot1,
                F_osc(t_ns + 0.5 * dt, Fmean, amp, freq),
                h,
                b,
                c,
            )
            x_dot3, y_dot3 = L96_2t_x_dot_y_dot(
                X + 0.5 * dt * x_dot2,
                Y + 0.5 * dt * y_dot2,
                F_osc(t_ns + 0.5 * dt, Fmean, amp, freq),
                h,
                b,
                c,
            )
            x_dot4, y_dot4 = L96_2t_x_dot_y_dot(
                X + dt * x_dot3,
                Y + dt * y_dot3,
                F_osc(t_ns + dt, Fmean, amp, freq),
                h,
                b,
                c,
            )

            X = X + (dt / 6.0) * ((x_dot1 + x_dot4) + 2.0 * (x_dot2 + x_dot3))
            Y = Y + (dt / 6.0) * ((y_dot1 + y_dot4) + 2.0 * (y_dot2 + y_dot3))

        x_hist[n + 1] = X
        y_hist[n + 1] = Y
        time[n + 1] = t_0 + si * (n + 1)

    return x_hist, y_hist, time
