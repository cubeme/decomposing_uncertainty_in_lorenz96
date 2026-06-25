"""Plot diagnostics for fitted baseline parameterizations."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly

from parameterization.utils.helpers import compute_ar_p_noise
from plotting.fit.distribution import (
    plot_x_u_distribution_with_polynomial_fit,
    plot_x_u_distribution_with_stochastic_polynomial_fit,
)
from plotting.fit.time_series import plot_residuals_with_noise

# ==============================================================================
# FUNCTION TO PLOT EVERYTHING
# THIS IS THE ONLY FUNCTION CALLED OUTSIDE OF THIS MODULE.
# ==============================================================================


def plot_all(
    x,
    t,
    u,
    coefs,
    rhos,
    sigmas,
    poly_order,
    seed,
    base_dir: Optional[str] = None,
):
    """Call all relevant plotting functions."""
    if base_dir is not None and not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Turn interactive mode off
    plt.ioff()

    if coefs is None:
        return

    plot_x_u_distribution_with_polynomial_fit(
        x,
        u,
        coefs,
        poly_label=f"$P_{poly_order}(X_k)$",
        output_folder=base_dir,
    )

    for rho, sigma in zip(rhos, sigmas):
        rho = np.asarray(rho)
        ar_order = rho.size

        # Plot residuals with AR(1) noise
        residuals = u - poly.polyval(x, coefs)
        ar_p_noise = compute_ar_p_noise(
            rho, sigma, residuals.shape[0], k=x.shape[1], seed=seed
        )
        # Window depends on sampling interval. Here we assume si=0.005
        plot_window = 2000
        plot_residuals_with_noise(
            residuals,
            ar_p_noise,
            t,
            time_start=max(residuals.shape[0] - plot_window - 1, 0),
            time_end=residuals.shape[0] - 1,
            output_folder=base_dir,
            ar_order=ar_order,
        )

        plot_x_u_distribution_with_stochastic_polynomial_fit(
            x,
            u,
            coefs,
            rho,
            sigma,
            poly_label=f"$P_{poly_order}(X_k) + AR({ar_order})$",
            output_folder=base_dir,
            seed=seed,
            ar_order=ar_order,
        )
