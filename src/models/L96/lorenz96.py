"""Define the fully resolved two-scale Lorenz '96 model."""

import numpy as np

from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)
from models.L96.time_stepping_rk4 import (
    run_rk4_const,
    run_rk4_linear,
    run_rk4_oscillating,
)


class L96:
    """
    Class for the two time-scale Lorenz '96 model.
    This model simulates a system with slow (X) and fast (Y) variables,
    coupled through a set of differential equations.

    Attributes:
        x (numpy.ndarray): Current state or initial conditions for the slow
            variables (X).
        y (numpy.ndarray): Current state or initial conditions for the fast
            variables (Y).
        f (float): Forcing term for the slow variables.
        h (float): Coupling coefficient between the slow and fast variables.
        b (float): Ratio of amplitudes between the fast and slow variables.
        c (float): Time-scale ratio between the fast and slow variables.
        t (float): Current time or initial time.
        k (int): Number of slow variables (X).
        j (int): Number of fast variables (Y) per slow variable.
    """

    def __init__(
        self,
        K,
        J,
        h=1,
        b=10,
        c=10,
        F_schedule=None,
        t=0,
        y_scale=1.0,
        seed=17,
    ):
        """
        Initialize the Lorenz '96 model with the given parameters.

        Args:
            K (int): Number of slow variables (X).
            J (int): Number of fast variables (Y) per slow variable.
            h (float): Coupling coefficient. Default is 1.
            b (float): Ratio of amplitudes. Default is 10.
            c (float): Time-scale ratio. Default is 10.
            F_schedule (ForcingSchedule): Forcing schedule for the slow variables.
                Default is ConstantForcingSchedule with F=18.
                Options: ConstantForcingSchedule, LinearForcingSchedule,
                OscillatingForcingSchedule.
            t (float): Initial time. Default is 0.
            seed (int): Random seed for reproducibility. Default is 17.
        """
        np.random.seed(seed)

        self.f_schedule = (
            ConstantForcingSchedule(18.0) if F_schedule is None else F_schedule
        )
        self.h, self.b, self.c = h, b, c
        self.y_scale = y_scale
        # Initialize system state
        self.x, self.y, self.t = (
            b * np.random.standard_normal((K,)),
            y_scale * np.random.standard_normal((K * J,)),
            t,
        )
        self.k, self.j = K, J
        self.jk = J * K  # Total number of fast variables

    def set_state(self, x_init, y_init, transform_fast=False):
        """
        Set the initial state of the Lorenz '96 model.

        Args:
            x_init (numpy.ndarray): Initial conditions for the slow variables (X).
            y_init (numpy.ndarray): Initial conditions for the fast variables (Y).
            transform_fast (bool, optional): If True, applies the y_scale factor
                to the fast variables. Default is False.
        """
        self.x = x_init.copy()
        if transform_fast:
            self.y = self.y_scale * y_init
        else:
            self.y = y_init.copy()

    def __repr__(self):
        return f"L96: K={self.k} J={self.j} schedule={self.f_schedule} h={self.h} b={self.b} c={self.c}, y_scale={self.y_scale}"

    def __str__(self):
        return self.__repr__() + f"\n X={self.x} \nY={self.y} \nt={self.t}"

    def spin_up(self, si, spin_up_time, dt=0.001):
        """
        Spin up the Lorenz '96 model for a specified duration to reach a
        statistically steady state. The forcing schedule is used during the spin-up.

        Args:
            si (float): Sampling interval (time increment for each step).
            spin_up_time (float): Total time to spin up the model.
            dt (float, optional): Time step for numerical integration. Default
                is 0.001.
        """

        # Number of time steps
        nt = int(spin_up_time / si)

        t_0 = self.t

        s = self.f_schedule
        if isinstance(s, ConstantForcingSchedule):
            F = s.F
        elif isinstance(s, LinearForcingSchedule):
            F = s.F0  # Use initial forcing value during spin-up
        elif isinstance(s, OscillatingForcingSchedule):
            F = s.Fmean  # Use mean forcing value during spin-up
        else:
            raise TypeError(f"Unknown forcing schedule type: {type(s)}")
        x_hist, y_hist, _ = run_rk4_const(
            nt, dt, si, t_0, self.x, self.y, self.h, self.c, self.b, F
        )

        # Set final state as new initial conditions and reset time to zero
        self.x, self.y, self.t = x_hist[-1], y_hist[-1], 0

    def run(
        self,
        si,
        total_time,
        dt=0.001,
        store=False,
    ):
        """
        Run the Lorenz '96 model for a total time `total_time`, sampling at intervals of `si`.

        Args:
            si (float): Sampling interval (time increment for each step).
            total_time (float): Total simulation time.
            dt (float, optional): Time step for numerical integration. Default
            store (bool, optional): If True, stores the final state as the
                initial conditions for the next run.

        Returns:
            tuple:
                - x_hist (numpy.ndarray): History of the slow variables (X)
                    over time.
                - y_hist (numpy.ndarray): History of the fast variables (Y)
                    over time.
                - time (numpy.ndarray): Array of time points corresponding to
                    the simulation.
        """

        # Number of time steps
        nt = int(total_time / si)

        t_0 = self.t

        s = self.f_schedule
        if isinstance(s, ConstantForcingSchedule):
            x_hist, y_hist, time = run_rk4_const(
                nt, dt, si, t_0, self.x, self.y, self.h, self.c, self.b, s.F
            )
        elif isinstance(s, LinearForcingSchedule):
            x_hist, y_hist, time = run_rk4_linear(
                nt,
                dt,
                si,
                t_0,
                self.x,
                self.y,
                self.h,
                self.c,
                self.b,
                s.F0,
                s.F1,
                s.t0,
                s.t1,
            )
        elif isinstance(s, OscillatingForcingSchedule):
            x_hist, y_hist, time = run_rk4_oscillating(
                nt,
                dt,
                si,
                t_0,
                self.x,
                self.y,
                self.h,
                self.c,
                self.b,
                s.Fmean,
                s.amp,
                s.freq,
            )
        else:
            raise TypeError(f"Unknown forcing schedule type: {type(s)}")

        if store:
            self.x, self.y, self.t = x_hist[-1], y_hist[-1], time[-1]

        return x_hist, y_hist, time
