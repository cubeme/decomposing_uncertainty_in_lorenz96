"""Provide shared plotting helpers and style settings."""

import matplotlib.pyplot as plt
from absl import logging


def empty_fig_on_failure(func):
    """Decorator for individual plot functions to return empty fig on failure."""

    def applicator(*args, **kwargs):
        # noinspection PyBroadException
        try:
            return func(*args, **kwargs)
        except Exception as e:  # pylint: disable=bare-except
            logging.warning("Plot failed. Caught exception: %s", repr(e))
            return plt.figure()

    return applicator


def save_plot(figure: plt.Figure, path: str):
    """Store a figure in a given location on disk."""
    if path is not None:
        figure.savefig(path, bbox_inches="tight", format="pdf")
        plt.close(figure)
