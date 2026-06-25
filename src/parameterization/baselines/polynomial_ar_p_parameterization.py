"""Define a polynomial parameterization with autoregressive residuals."""

import pickle
from pathlib import Path

import numpy as np
import numpy.polynomial.polynomial as poly

from parameterization.base_parameterization import BaseParameterization


class PolynomialARpParameterization(BaseParameterization):
    """
    Polynomial parameterization with an autoregressive AR(p) noise model.

    The parameterization is defined as:
        U_p = P(X) + e(t),
    where P(X) is a polynomial function of the state variables X, and e(t) is an AR(p) noise process:
        e(t) = sum_{i=1..p} rho_i * e(t-i) + sigma * z(t),
    where z(t) is Gaussian white noise.
    """

    def __init__(
        self,
        coefs: np.ndarray,
        rho: float | np.ndarray,
        sigma: float,
        seed: int = 0,
    ):
        """
        Initialize the parameterization.

        Args:
            coefs (np.ndarray): Polynomial coefficients.
            rho (float | np.ndarray): Autoregressive coefficients for the AR(p) noise process.
            sigma (float): Standard deviation of the noise term in the AR(p) process.
            seed (int, optional): Random seed for reproducibility. Default is 17.
        """
        self.coefs = coefs
        self.rho = np.asarray(rho, dtype=float).reshape(-1)
        self.sigma = float(sigma)
        self.seed = seed

        # Validate rho values
        if self.rho.size == 1:
            rho = float(self.rho[0])
            if not (-1.0 <= rho <= 1.0):
                raise ValueError("rho must satisfy -1 <= rho <= 1.")

        if self.rho.size < 1:
            raise ValueError("rho must have length >= 1.")

        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError("sigma must be a finite nonnegative float.")

        self.rg = np.random.default_rng(seed=seed)

        # Initialize noise value standard Gaussian noise
        self.noise = self.rg.normal(0, 1, size=(1,))

        # Initialize noise history (newest first): [e_t, e_{t-1}, ...]
        self._noise_hist = []

    def update(self):
        p = int(self.rho.size)

        # Ensure noise has correct shape and initialize history if needed
        if len(self._noise_hist) != p or (
            len(self._noise_hist) > 0 and self._noise_hist[0].shape != self.noise.shape
        ):
            self._noise_hist = [self.noise.copy() for _ in range(p)]

        z = self.rg.normal(0, 1, size=self.noise.shape)

        # AR(p): e(t) = sum_{i=1..p} rho_i e(t-i) + sigma * z(t)
        e_next = 0.0
        for i in range(p):
            e_next = e_next + float(self.rho[i]) * self._noise_hist[i]
        e_next = e_next + self.sigma * z

        # shift history: new newest first
        self._noise_hist = [e_next] + self._noise_hist[: p - 1]
        self.noise = self._noise_hist[0]

    def predict(self, x, F=None):
        x_shape = np.shape(x)
        k = x_shape[-1] if x_shape else 1
        if self.noise.shape != (k,):
            self.noise = self.rg.normal(0, 1, size=(k,))
            # reset history on shape change
            p = int(self.rho.size)
            self._noise_hist = [self.noise.copy() for _ in range(p)]
        return poly.polyval(x, self.coefs) + self.noise

    def save(self, save_file):
        if not str(save_file).endswith(".p"):
            raise ValueError("Output file must be a pickle file (.p).")

        d = {
            "coefs": self.coefs,
            "rho": self.rho,
            "sigma": self.sigma,
            "seed": self.seed,
            "noise": self.noise,
            "noise_hist": self._noise_hist,
        }

        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(d, f)


def load_polynomial_ar_p_parameterization(load_file):
    with open(load_file, "rb") as f:
        d = pickle.load(f)

    p = PolynomialARpParameterization(d["coefs"], d["rho"], d["sigma"], d["seed"])
    p.noise = d["noise"]
    p._noise_hist = d.get(
        "noise_hist", [p.noise.copy() for _ in range(int(p.rho.size))]
    )
    return p
