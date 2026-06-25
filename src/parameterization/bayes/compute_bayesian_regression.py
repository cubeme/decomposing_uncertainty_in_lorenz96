"""Fit Bayesian polynomial regression models."""

import multiprocessing as mp

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _poly_eval(x, b):
    """
    Polynomial evaluation: sum_i b[i] * x**i

    Args:
        x: PyTensor variable (any shape)
        b: 1D PyTensor variable (coeffs, ascending order)
    """
    res = pt.zeros_like(x)

    for i, c in enumerate(b):
        res = res + c * (x**i)
    return res


def fit_bayesian_regression(
    x, u, poly_order, chains=4, draws=1000, tune=2000, return_samples=300
):
    """
    Fit Bayesian polynomial regression using PyMC.

    Args:
        x (np.ndarray): Input features, shape (N, k).
        u (np.ndarray): Target values, shape (N, k).
        poly_order (int): Order of the polynomial.

    Returns:
        b_flat: Posterior samples of the polynomial coefficients.
    """
    # Avoid ConnectionResetError in subprocesses
    # See: https://github.com/pymc-devs/pymc/issues/7354
    mp.set_start_method("spawn", force=True)
    rng = np.random.default_rng()

    with pm.Model() as model:
        # Coefficients for each degree and feature
        b = pm.Normal("b", mu=0, sigma=1, shape=(poly_order + 1))  # (p+1)

        sigma = pm.HalfCauchy("sigma", beta=10)

        mu = pm.Deterministic("mu", _poly_eval(x, b))  # (N, k)

        likelihood = pm.Normal("u_pred", mu=mu, sigma=sigma, observed=u)

        trace = pm.sample(chains=chains, draws=draws, tune=tune)

    b_samples = trace.posterior["b"]
    b_flat = b_samples.stack(sample=("chain", "draw")).transpose("sample", "b_dim_0")

    choice_idx = rng.choice(b_flat.shape[0], size=return_samples, replace=False)
    b_flat = b_flat[choice_idx]

    return b_flat.to_numpy()
