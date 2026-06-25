"""Define the parameterized coarse-grained Lorenz '96 model."""

from typing import Callable, Tuple

import numpy as np

from models.forcing_schedule import (
    ConstantForcingSchedule,
    forcing_at,
)
from models.GCM.time_stepping import RK2
from models.helpers import L96_eq1_x_dot
from parameterization.base_parameterization import BaseParameterization


class GCM:
    """
    General Circulation Model (GCM) of the Lorenz '96 model using manual time-stepping methods.
    """

    def __init__(
        self,
        parameterization: BaseParameterization,
        F_schedule=None,
    ):
        """
        Initialize GCM instance.

        Args:
            parameterization (BaseParameterization): Parameterization to modify
                the RHS of the tendency equation. Defaults to a zero polynomial.
            F_schedule (ForcingSchedule): Forcing schedule for the Lorenz '96 model.
        """
        self.f_schedule = (
            ConstantForcingSchedule(18.0) if F_schedule is None else F_schedule
        )
        self.parameterization = parameterization

    def _forcing_at(self, t: float) -> float:
        """
        Get the forcing value at time t.
        """
        return forcing_at(self.f_schedule, t)

    def rhs(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute the right-hand side (RHS) of the tendency equation with parameterization.

        Args:
            x (np.ndarray): Current state of the large-scale variables.

        Returns:
            np.ndarray: The computed RHS of the parameterized tendency
                equation.
        """
        F_t = self._forcing_at(t)
        return L96_eq1_x_dot(x, F_t) - self.parameterization.predict(x, F_t)

    def __call__(
        self,
        x_init: np.ndarray,
        si: float,
        total_time: float,
        dt: float,
        time_stepping_func: Callable = RK2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates the system forward in time.

        Args:
            x_init (np.ndarray): Initial conditions for the large-scale state
                variables. Shape: (K,).
            dt (float): Time step for numerical integration.
            si (float): Sampling interval (time increment for each step).
            total_time (float): Total simulation time.
            time_stepping_func (Callable): Time-stepping function (e.g., RK4).
                Defaults to 'euler_forward'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - hist (np.ndarray): History of the large-scale state over
                    time. Shape: (nt + 1, K).
                - time (np.ndarray): Array of time points corresponding to the
                    simulation. Shape: (nt + 1,).
        """
        # Number of time steps
        nt = int(total_time / si)

        # Compute number of integration steps
        if si < dt:
            dt, ns = si, 1
        else:
            ns = int(si / dt + 0.5)
            assert abs(ns * dt - si) < 1e-14, (
                f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"
            )

        # Initialize history array for storing the state variables
        hist = np.full((nt + 1, len(x_init)), np.nan)
        time = np.zeros((nt + 1))

        x = x_init.copy()
        hist[0] = x

        for n in range(nt):
            t_n = si * n
            for s in range(ns):
                t_ns = t_n + s * dt
                x = time_stepping_func(self.rhs, dt, x, t_ns)

            self.parameterization.update()
            hist[n + 1], time[n + 1] = x, si * (n + 1)

        return hist, time
