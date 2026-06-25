"""Plot Lorenz '96 time series."""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from absl import logging

from plotting.helpers import empty_fig_on_failure


@empty_fig_on_failure
def plot_l96_x_with_random_y(
    x, y, t, time_end, time_start=0, figsize=(10, 16), columns=2, output_folder=""
):
    """
    Plots time series for Lorenz '96 slow variables (X) and randomly selected fast variables (Y).

    This function generates a time series plot for each slow variable (X) and a randomly selected
    corresponding fast variable (Y) from the Lorenz '96 model.

    Args:
        x_true (numpy.ndarray): Array of slow variables (X) with shape (time, k).
        y_true (numpy.ndarray): Array of fast variables (Y) with shape (time, jk).
        t (numpy.ndarray): Array of time points with shape (time,).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot.
            Default is 0.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
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
    kj = y.shape[-1]
    j = kj // k

    y = y.reshape((y.shape[0], k, j))

    sns.set_theme()

    time_slice = t[time_start:time_end]

    nrows = math.ceil(k / columns)
    fig, axes = plt.subplots(nrows, columns, figsize=figsize, sharex=True)

    # ensure axes is 2D
    axes = np.atleast_2d(axes)

    for i in range(k):
        r = i // columns
        c = i % columns
        ax = axes[r, c]

        # Select random fast variable
        random_y_index = np.random.randint(0, j)
        y_index = f"Y_{{{i},{random_y_index}}}"

        # Plot
        ax.plot(time_slice, x[time_start:time_end, i], label=f"$X_{i}(t)$")
        ax.plot(
            time_slice,
            y[time_start:time_end, i, random_y_index],
            label=f"${y_index}(t)$",
        )

        ax.set_xlabel("Time $t$")
        ax.set_title(f"$X_{i}$ and ${y_index}(t)$")
        ax.legend()

    # turn off unused subplots
    for idx in range(k, nrows * columns):
        r = idx // columns
        c = idx % columns
        axes[r, c].axis("off")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = "l96_training_time_series.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logging.info("X/Y time series plot saved successfully")

    return fig
