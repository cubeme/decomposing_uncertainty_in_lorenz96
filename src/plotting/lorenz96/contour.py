"""Plot contour views of Lorenz '96 simulations."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging

from plotting.helpers import empty_fig_on_failure


@empty_fig_on_failure
def plot_l96_x_and_y(
    x,
    y,
    t,
    time_end,
    time_start=0,
    cmap="viridis",
    output_folder="",
):
    """
    Plots a contour plot for Lorenz '96 slow variables (X) and fast variables (Y).

    This function generates two contour plots in one figure:
    - The first plot shows the evolution of the slow variables X over time.
    - The second plot shows the evolution of the corresponding fast variables Y over time.

    Args:
        x_true (numpy.ndarray): Array of slow variables (X) with shape (time, k).
        y_true (numpy.ndarray): Array of fast variables (Y) with shape (time, jk).
        t (numpy.ndarray): Array of time points.
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot.
            Default is 0.
        cmap (str, optional): Colormap to use for the contour plots. Default is 'viridis'.
        output_folder (str, optional): Path to the folder where the plot will be saved as a PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.
        config (dict, optional): Configuration dictionary containing simulation parameters.
            Used for naming the output file if 'output_folder' is given. Defaults to None.
            Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 'total_time' (float): Total simulation time.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    k = x.shape[-1]
    jk = y.shape[-1]
    j = jk // k

    sns.set_theme()
    fig = plt.figure(figsize=(10, 6))

    time_slice = t[time_start:time_end]

    # Plot the slow variables (X)
    plt.subplot(121)  # nrows, ncols, index
    plt.contourf(np.arange(k), time_slice, x[time_start:time_end, :], cmap=cmap)
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel("t")
    plt.yticks(time_slice[:: max(1, len(time_slice) // 10)])
    plt.title("$X_k(t)$")

    # Plot the fast variables (Y)
    plt.subplot(122)
    plt.contourf(
        np.arange(jk) / j,
        time_slice,
        y[time_start:time_end, :],
        levels=np.linspace(-1, 1, 10),
        cmap=cmap,
    )
    plt.xlabel("k+j/J")
    plt.ylabel("t")
    plt.yticks(time_slice[:: max(1, len(time_slice) // 10)])
    plt.title("$Y_{j,k}(t)$")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = "l96_training2d.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logging.info("L96 contour plot saved successfully.")

    return fig
