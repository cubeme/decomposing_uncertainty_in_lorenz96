"""Plot contour diagnostics for parameterization fits."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging

from plotting.helpers import empty_fig_on_failure

label_dict = {
    "full": {
        "label": "Full L96",
    },
    "no_param": {
        "label": "No parameterization",
    },
    "det_param": {
        "label": "Deterministic parameterization",
    },
    "stoch_param": {
        "label": "Stochastic parameterization",
    },
    "bayes_param": {
        "label": "Bayesian parameterization",
    },
}

# Define contour plot limits and levels
LIMITS = {
    "vmin": -12,
    "vmax": 12,
    "levels": np.linspace(-12, 12, 12),
    "extend": "both",
}


@empty_fig_on_failure
def plot_gcm_contour_comparison(
    t,
    x_data,
    time_end,
    time_start=0,
    cmap="viridis",
    limits=None,
    max_ncols=3,
    output_folder="",
):
    """
    Plot contour plots for multiple datasets.

    Args:
        t (np.ndarray): Time array, shape (time,).
        data_dict (dict[str, np.ndarray]): Dictionary mapping subplot titles to data arrays (time, k).
        time_end (int): Last time index to plot (exclusive).
        time_start (int): First time index (inclusive).
        cmap (str): Colormap. Default is 'viridis'.
        limits (dict | None): Contour kwargs, e.g. {"vmin": -12, "vmax": 12, "levels": np.linspace(-12,12,12)}.
        max_ncols (int): Number of columns of subplots.
        output_path (str): If not empty, save the figure to this path.

    Returns:
        matplotlib.figure.Figure
            The generated figure object.
    """
    sns.set_theme(style="ticks")

    # assume all arrays have same shape
    k = next(iter(x_data.values())).shape[-1]
    range_k = np.arange(k)
    t_slice = t[time_start:time_end]

    # Define contour plot limits and levels
    limits = LIMITS if limits is None else limits

    # Plot max max_ncols subplots per row
    nplots = len(x_data)
    if nplots <= max_ncols:
        ncols = nplots
    else:
        ncols = max_ncols
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 8 * nrows), squeeze=False
    )

    for i, (key, x) in enumerate(x_data.items()):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        cf = ax.contourf(
            range_k, t_slice, x[time_start:time_end, :], cmap=cmap, **limits
        )
        ax.set_xlabel("$K$")
        ax.set_ylabel("$t$")
        ax.set_title(label_dict.get(key, {}).get("label", key))
        plt.colorbar(cf, ax=ax)

    # Hide unused axes
    for j in range(nplots, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        _save_plot("gcm_2d_comparison", output_folder)
        logging.info("GCM contour comparison plot saved successfully.")

    return fig


@empty_fig_on_failure
def plot_gcm_contour_difference(
    t,
    x_full,
    x_data,
    time_end,
    time_start=0,
    limits=None,
    max_ncols=3,
    cmap="viridis",
    output_folder="",
):
    """
    Plots contour differences for GCM simulations with deterministic and
    stochastic parameterizations.

    This function generates three contour plots:
    - The first plot shows the full Lorenz '96 model (x_full).
    - The second plot shows the difference between the full L96 model and the GCM
      with deterministic parameterization.
    - The third plot shows the difference between the full L96 model and the GCM
      with stochastic parameterization.

    Args:
        t (numpy.ndarray): Array of time points with shape (time,).
        x_full (numpy.ndarray): Array of variables from the full Lorenz '96 model
            with shape (time, k).
        x_det_param (numpy.ndarray): Array of variables from GCM with deterministic
            parameterization with shape (time, k).
        x_stoch_param (numpy.ndarray): Array of variables from GCM with stochastic
            parameterization with shape (time, k).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot.
            Default is 0.
        cmap (str, optional): Colormap to use for the contour plots. Default is 'viridis'.
        output_path (str, optional): Path to the folder where the plot will be saved as a PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    sns.set_theme(style="ticks")

    k = x_full.shape[-1]
    range_k = np.arange(k)
    t_slice = t[time_start:time_end]
    x_full_slice = x_full[time_start:time_end, :]
    # Define contour plot limits and levels
    limits = LIMITS if limits is None else limits

    # Prepare data to plot: full + differences
    data_to_plot = {"full": x_full_slice}
    for key, x in x_data.items():
        data_to_plot[key] = x_full_slice - x[time_start:time_end, :]

    # Plot max max_ncols subplots per row
    nplots = len(data_to_plot)
    if nplots <= max_ncols:
        ncols = nplots
    else:
        ncols = max_ncols
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 8 * nrows), squeeze=False
    )

    for i, (key, arr) in enumerate(data_to_plot.items()):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        cf = ax.contourf(range_k, t_slice, arr, cmap=cmap, **limits)
        ax.set_xlabel("$k$")
        ax.set_ylabel("$t$")

        if key == "full":
            title = "Full L96"
        else:
            title = f"L96 - {label_dict.get(key, {}).get('label', key)}"

        ax.set_title(title)
        plt.colorbar(cf, ax=ax)

    # Hide unused axes
    for j in range(nplots, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        _save_plot("gcm_2d_difference", output_folder)
        logging.info("GCM contour difference plot saved successfully.")

    return fig


def _save_plot(plot_type, output_folder):
    f_name = f"{plot_type}.pdf"
    save_path = os.path.join(output_folder, f_name)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
