"""Plot diagnostics for fitted Bayesian parameterizations."""

import os
from typing import Optional

import matplotlib.pyplot as plt

from plotting.fit.distribution import (
    plot_x_u_distribution_with_bayesian_polynomial_fit,
)


# ==============================================================================
# FUNCTION TO PLOT EVERYTHING
# THIS IS THE ONLY FUNCTION CALLED OUTSIDE OF THIS MODULE.
# ==============================================================================


def plot_all(
    x,
    u,
    bayes_coefs,
    poly_order,
    base_dir: Optional[str] = None,
):
    """Call all relevant plotting functions."""
    if base_dir is not None and not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Turn interactive mode off
    plt.ioff()

    if bayes_coefs is None:
        return

    plot_x_u_distribution_with_bayesian_polynomial_fit(
        x,
        u,
        bayes_coefs,
        poly_label=f"$P_{poly_order}(X_k)_i$",
        output_path=base_dir,
    )
