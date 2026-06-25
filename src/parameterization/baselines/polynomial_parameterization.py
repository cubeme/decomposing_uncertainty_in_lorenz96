"""Define deterministic and Bayesian polynomial parameterizations."""

import pickle
from pathlib import Path

import numpy as np
import numpy.polynomial.polynomial as poly

from parameterization.base_parameterization import BaseParameterization


class PolynomialParameterization(BaseParameterization):
    """
    Deterministic polynomial parameterization.

    The parameterization is defined as:
        U_p = P(X),
    where P(X) is a polynomial function of the state variables X.
    """

    def __init__(self, coefs: np.ndarray):
        """
        Initialize the parameterization.

        Args:
            coefs (np.ndarray): Polynomial coefficients.
        """
        self.coefs = coefs

    def update(self):
        pass

    def predict(self, x, F=None):
        return poly.polyval(x, self.coefs)

    def save(self, save_file):
        if not str(save_file).endswith(".p"):
            raise ValueError("Output file must be a pickle file (.p).")

        d = {
            "coefs": self.coefs,
        }

        Path(save_file).parent.mkdir(parents=True, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(d, f)


def load_polynomial_parameterization(load_file):
    with open(load_file, "rb") as f:
        d = pickle.load(f)

    p = PolynomialParameterization(d["coefs"])

    return p
