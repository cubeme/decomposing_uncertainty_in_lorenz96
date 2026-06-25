"""Plot time-series diagnostics for parameterization fits."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging

from plotting.helpers import empty_fig_on_failure

colorblind_palette_10 = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#882255",  # wine
    "#D55E00",  # vermillion
    "#999999",  # grey
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#44AA99",  # teal
]


@empty_fig_on_failure
def plot_residuals_with_noise(
    residuals,
    ar_p_noise,
    t,
    time_end,
    time_start=0,
    ar_order=1,
    figsize=(10, 8),
    columns=2,
    output_folder="",
):
    """
    Plots residuals and AR(p) noise time series for Lorenz '96 model.

    This function generates a time series plot for each residual and its corresponding AR(p) noise.

    Args:
        residuals (numpy.ndarray): Array of residuals with shape (time, k).
        ar_p_noise (numpy.ndarray): Array of AR(p) noise with shape (time, k).
        t (numpy.ndarray): Array of time points with shape (time,).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot.
            Default is 0.
        ar_order (int, optional): Order of the AR process. Default is 1.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    sns.set_theme()

    k = residuals.shape[-1]
    nrows = math.ceil(k / columns)

    time_slice = t[time_start:time_end]

    fig, axes = plt.subplots(nrows, columns, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for i in range(k):
        r = i // columns
        c = i % columns
        ax = axes[r, c]

        ax.plot(
            time_slice,
            residuals[time_start:time_end, i],
            label=f"$Res_{i}(t)$",
        )
        ax.plot(
            time_slice,
            ar_p_noise[time_start:time_end, i],
            label=f"$AR({ar_order})_{i}(t)$",
        )

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$X_{i}$ residuals + AR({ar_order}) noise")
        ax.legend(fontsize=7, loc="upper right")

    # hide unused axes
    for idx in range(k, nrows * columns):
        r = idx // columns
        c = idx % columns
        axes[r, c].axis("off")

    plt.tight_layout(pad=2.0)

    if output_folder != "":
        _save_plot(f"residuals_ar{ar_order}", output_folder)
        logging.info(
            "Residuals with AR(%d) noise time series plot saved successfully", ar_order
        )

    return fig


@empty_fig_on_failure
def plot_gcm_time_series_comparison(
    t,
    x_data,
    time_end,
    time_start=0,
    figsize=(12, 16),
    columns=2,
    output_folder="",
):
    """
    Plots time series comparison for GCM simulations with different parameterizations.

    This function generates a time series plot for each variable in the GCM simulation, comparing:
    - Full Lorenz '96 model (x_full)
    - No parameterization (x_no_param)
    - Deterministic parameterization (x_det_param)
    - Stochastic parameterization (x_stoch_param)

    Args:
        t (numpy.ndarray): Array of time points with shape (time,).
        x_full (numpy.ndarray): Array of variables from the full Lorenz '96 model
            with shape (time, k).
        x_no_param (numpy.ndarray): Array of variables from GCM with no
            parameterization with shape (time, k).
        x_det_param (numpy.ndarray): Array of variables from GCM deterministic
            parameterization with shape (time, k).
        x_stoch_param (numpy.ndarray): Array of variables from GCM stochastic
            parameterization with shape (time, k).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot.
            Default is 0.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.


    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set theme
    sns.set_theme()

    style_dict = {
        "full": {"label": "Full L96", "color": colorblind_palette_10[0]},
        "no_param": {"label": "No parameterization", "color": colorblind_palette_10[1]},
        "det_param": {
            "label": "Deterministic parameterization",
            "color": colorblind_palette_10[2],
        },
        "stoch_param": {
            "label": "Stochastic parameterization",
            "color": colorblind_palette_10[3],
        },
        "bayes_param": {
            "label": "Bayesian parameterization",
            "color": colorblind_palette_10[4],
        },
    }

    k = x_data["full"].shape[-1]
    nrows = math.ceil(k / columns)

    time_slice = t[time_start:time_end]

    fig, axes = plt.subplots(nrows, columns, figsize=figsize, sharex=True)
    axes = np.atleast_2d(axes)

    for i in range(k):
        r = i // columns
        c = i % columns
        ax = axes[r, c]

        for key, x in x_data.items():
            ax.plot(
                time_slice,
                x[time_start:time_end, i],
                label=style_dict[key]["label"],
                color=style_dict[key]["color"],
            )

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$X_{i}$")

    # hide unused subplots
    for idx in range(k, nrows * columns):
        r = idx // columns
        c = idx % columns
        axes[r, c].axis("off")

    # collect handles/labels from first axis
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # place legend on the right of the whole figure
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
    )

    plt.tight_layout()

    if output_folder != "":
        _save_plot("gcm_time_series", output_folder)
        logging.info("GCM time series comparison plot saved successfully.")

    return fig


def _save_plot(
    plot_type,
    output_folder,
):
    f_name = f"{plot_type}.pdf"
    save_path = os.path.join(output_folder, f_name)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
