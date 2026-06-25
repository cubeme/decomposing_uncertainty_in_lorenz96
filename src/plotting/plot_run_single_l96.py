"""Plot diagnostics for a single Lorenz '96 simulation."""

import os
from typing import Optional

import matplotlib.pyplot as plt
from plotting.lorenz96.contour import plot_l96_x_and_y
from plotting.lorenz96.time_series import plot_l96_x_with_random_y


def plot_all(x, y, t, plot_start, base_dir: Optional[str] = None):
    """Call all relevant plotting functions.
    Args:
        results: The dictionary containing all results from the run.
        base_dir: The path where to store the figures. If `None` don't save the
              figures to disk.
        suffix: An optional suffix to each filename stored by this function.
    """
    if base_dir is not None and not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Turn interactive mode off
    plt.ioff()
    # Window depends on sampling interval. Here we assume si=0.005
    plot_window = 2000

    plot_l96_x_and_y(
        x,
        y,
        t,
        time_start=plot_start,
        time_end=plot_start + plot_window,
        output_folder=base_dir,
    )
    plot_l96_x_with_random_y(
        x,
        y,
        t,
        time_start=plot_start,
        time_end=plot_start + plot_window,
        output_folder=base_dir,
    )
