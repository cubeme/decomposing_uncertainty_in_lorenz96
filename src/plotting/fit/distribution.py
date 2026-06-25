"""Plot distribution diagnostics for parameterization fits."""

import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import seaborn as sns
from absl import logging

from plotting.helpers import empty_fig_on_failure


@empty_fig_on_failure
def plot_x_u_distribution_with_polynomial_fit(
    x, u, coefs, poly_label="$poly(X_k)$", output_folder=""
):
    """
    Plot the joint distribution of X and U with a polynomial fit.

    This function generates a 2D histogram of the slow variables (X) and the coupling term (U),
    and overlays a polynomial fit on the histogram.

    Args:
        x (numpy.ndarray): Array of slow variables (X) with shape (n_samples, k).
        u (numpy.ndarray): Array of coupling terms (U) with shape (n_samples, k).
        coefs (numpy.ndarray): Coefficients of the polynomial fit.
        poly_label (str, optional): Label for the polynomial fit in the legend.
            Default is "$poly(X_k)$".
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set style
    sns_style = "viridis"
    sns.set_theme(style="ticks", palette=sns_style, color_codes=True)

    fig = plt.figure()

    # 2D histogram of X vs U
    _plot_histogram(x, u, sns_style=sns_style)

    # Fits from polynomials
    plot_x = np.linspace(x.flatten().min(), x.flatten().max(), 100)
    plt.plot(plot_x, poly.polyval(plot_x, coefs), color="xkcd:red", label=poly_label)

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        _save_plot("det_param", output_folder)
        logging.info("X/U density with polynomial fit plot saved successfully")

    return fig


def _simulate_ar_noise(rho, sigma_i, n=1000, burnin=200):
    # Cannot analytically derive sigma_e for AR(p) processes with p>1, but we can simulate noise and compute empirical sigma_e
    p = len(rho)
    noise = np.zeros(n + burnin)
    for t in range(p, n + burnin):
        noise[t] = np.dot(rho, noise[t - p : t][::-1]) + np.random.normal(0, sigma_i)
    return noise[burnin:]


@empty_fig_on_failure
def plot_x_u_distribution_with_stochastic_polynomial_fit(
    x,
    u,
    coefs,
    rho,
    sigma_i,
    poly_label="$poly(X)$",
    output_folder="",
    seed=None,
    ar_order=1,
):
    """
    Plot the joint distribution of X and U with a stochastic polynomial fit.

    This function generates a 2D histogram of the slow variables (X) and the coupling term (U),
    and overlays a stochastic polynomial fit on the histogram. The stochastic fit includes
    a polynomial component and an AR(p) noise component.

    Args:
        x (numpy.ndarray): Array of slow variables (X) with shape (n_samples, k).
        u (numpy.ndarray): Array of coupling terms (U) with shape (n_samples, k).
        coefs (numpy.ndarray): Coefficients of the polynomial fit.
        rho (float or np.ndarray): AR(p) coefficients for the noise model.
        sigma_i (float): Innovation sigma of the AR(p) model.
        poly_label (str, optional): Label for the polynomial fit in the legend.
            Default is "$poly(X_k) + AR(1)$".
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.
        ar_order (int, optional): Order of the AR process. Default is 1.
    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set style
    sns_style = "viridis"
    sns.set_theme(style="ticks", palette=sns_style, color_codes=True)

    fig = plt.figure()

    # 2D histogram of X vs U
    _plot_histogram(x, u, sns_style=sns_style)

    # Plot fits from polynomials + AR(p) noise
    plot_x = np.linspace(x.flatten().min(), x.flatten().max(), 100)
    plot_y = poly.polyval(plot_x, coefs)

    if ar_order == 1:
        sigma_e = sigma_i / (1 - rho**2) ** 0.5
    else:
        noise_samples = _simulate_ar_noise(rho, sigma_i, n=5000)
        sigma_e = np.std(noise_samples)

    plt.plot(
        plot_x,
        plot_y,
        color="xkcd:red",
        label=poly_label,
    )
    plt.fill_between(
        plot_x,
        plot_y - sigma_e,
        plot_y + sigma_e,
        color="xkcd:red",
        alpha=0.4,
        label=r"poly(X) $\pm \sigma_e$",
    )

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        _save_plot(f"stoch_param_ar{ar_order}", output_folder)
        logging.info(
            "X/U density with stochastic polynomial fit plot saved successfully"
        )

    return fig


@empty_fig_on_failure
def plot_x_u_distribution_with_bayesian_polynomial_fit(
    x, u, coefs, poly_label="$poly_i(X_k)$", output_path=""
):
    """
    Plot the joint distribution of X and U with a polynomial fit.

    This function generates a 2D histogram of the slow variables (X) and the coupling term (U),
    and overlays a polynomial fit on the histogram.

    Args:
        x (numpy.ndarray): Array of slow variables (X) with shape (n_samples, k).
        u (numpy.ndarray): Array of coupling terms (U) with shape (n_samples, k).
        coefs (numpy.ndarray): Coefficients of the polynomial fit.
        poly_label (str, optional): Label for the polynomial fit in the legend.
            Default is "$poly(X_k)$".
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set style
    sns_style = "rocket"
    sns.set_theme(style="ticks", palette=sns_style, color_codes=True)

    fig = plt.figure()

    # 2D histogram of X vs U
    _plot_histogram(x, u, sns_style=sns_style)

    # Fits from polynomials
    plot_x = np.linspace(x.flatten().min(), x.flatten().max(), 100)
    N, M, D = coefs.shape
    coefs = coefs.reshape(N * M, D)
    for idx in range(coefs.shape[0]):
        plt.plot(
            plot_x,
            poly.polyval(plot_x, coefs[idx]),
            color="xkcd:sky blue",
            label=poly_label if idx == 0 else None,  # only first line gets label
        )

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_path != "":
        _save_plot("bayesian_param", output_path)
        logging.info("X/U density with Bayesian polynomial fit plot saved successfully")

    return fig


def _plot_histogram(x, u, sns_style="rocket"):
    plt.hist2d(x.flatten(), u.flatten(), bins=40, density=True, cmap=sns_style)
    plt.xlabel("$X_k$")
    plt.ylabel(r"$U_k = \frac{hc}{b}\sum_j Y_{j,k}$")
    plt.colorbar(label="PDF")


def _save_plot(param, output_folder):
    f_name = f"{param}_fit.pdf"
    save_path = os.path.join(output_folder, f_name)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
