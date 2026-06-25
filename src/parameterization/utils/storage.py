"""Store and load fitted parameterization parameters."""

from pathlib import Path

import numpy as np


def save_poly_coefficients(out_path, coefs):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "coefs.npy", coefs)


def load_poly_coefficients(load_path):
    return np.load(Path(load_path) / "coefs.npy")


def save_ar_p_parameters(out_path, rho, sigma, ar_order=1):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / f"rho_{ar_order}.npy", np.asarray(rho, dtype=float))
    np.save(out_path / f"sigma_{ar_order}.npy", float(sigma))


def load_ar_p_parameters(load_path, ar_order=1):
    load_path = Path(load_path)
    rho_arr = np.asarray(np.load(load_path / f"rho_{ar_order}.npy"), dtype=float)
    sigma = float(np.load(load_path / f"sigma_{ar_order}.npy"))

    rho = float(rho_arr.reshape(-1)[0]) if rho_arr.size == 1 else rho_arr

    return rho, sigma


def save_bayesian_poly_coefficients(out_path, coefs):
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "bayesian_coefs.npy", coefs)


def load_bayesian_poly_coefficients(load_path):
    return np.load(Path(load_path) / "bayesian_coefs.npy")
