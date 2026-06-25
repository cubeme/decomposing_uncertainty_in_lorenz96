"""Initialize and execute Lorenz '96 simulations."""

import numpy as np
from absl import logging

from models.GCM.gcm import GCM
from models.helpers import (
    parse_time_stepping_func,
)
from models.L96.lorenz96 import L96
from parameterization.baselines.polynomial_ar_p_parameterization import (
    PolynomialARpParameterization,
)
from parameterization.baselines.polynomial_parameterization import (
    PolynomialParameterization,
)


def initialize_poly_param_gcm(config, coefs=None):
    if coefs is None:
        param = PolynomialParameterization(np.zeros(1))
    else:
        param = PolynomialParameterization(coefs)

    return GCM(param, F_schedule=config.f_schedule)


def initialize_poly_ar_p_param_gcm(config, coefs, rho, sigma):
    stoch_param = PolynomialARpParameterization(coefs, rho, sigma, config.seed)

    return GCM(stoch_param, F_schedule=config.f_schedule)


def run_gcm(gcm, init_conditions, config):
    logging.info("Run manual GCM for %d time steps...", config.total_time)

    return gcm(
        init_conditions,
        si=config.si,
        total_time=config.total_time,
        dt=config.dt,
        time_stepping_func=parse_time_stepping_func(config.time_stepping),
    )

def initialize_l96(config, seed=None):
    if seed is None:
        seed = config.seed

    m = L96(
        config.K,
        config.J,
        config.h,
        config.b,
        config.c,
        F_schedule=config.f_schedule,
        y_scale=config.y_scale,
        seed=seed,
    )

    if config.spin_up_time > 0:
        logging.info("Spin up L96...")
        # Spin up model and save final state as new initial state
        _ = m.spin_up(
            si=config.si,
            spin_up_time=config.spin_up_time,
            dt=config.dt,
        )

    return m


def run_l96(m, config, store=True):
    logging.info("Run L96 for %d time steps...", config.total_time)

    x, y, t = m.run(
        config.si,
        config.total_time,
        dt=config.dt,
        store=store,
    )

    return x, y, t
