"""Fit polynomial baseline parameters."""

import numpy.polynomial.polynomial as poly


def fit_deterministic_poly_coefs(x, u, poly_order):
    return poly.polyfit(x.flatten(), u.flatten(), poly_order)
