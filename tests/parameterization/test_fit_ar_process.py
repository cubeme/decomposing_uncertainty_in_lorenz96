import numpy as np
import pytest

from parameterization.utils.fit_ar_process import (
    fit_ar_p_ls_pooled,
    fit_ar_p_pooled,
    fit_baseline_poly_ar,
    fit_flow_latent_ar,
    is_stable_ar,
    project_to_stable_ar,
)

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------


def simulate_arp(rho, sigma, T=3000, K=3, seed=0):
    rho = np.asarray(rho).reshape(-1)
    p = len(rho)
    rng = np.random.default_rng(seed)
    x = np.zeros((T, K))
    for t in range(p, T):
        pred = np.zeros(K)
        for i in range(p):
            pred += rho[i] * x[t - i - 1]
        x[t] = pred + sigma * rng.normal(size=K)
    return x


# -------------------------------------------------------------------
# stability
# -------------------------------------------------------------------


def test_is_stable_ar():
    assert is_stable_ar(np.array([]))
    assert is_stable_ar(np.array([0.5]))
    assert not is_stable_ar(np.array([1.2]))


def test_project_to_stable_ar():
    rho = np.array([1.5, 1.7])
    rho2, ok = project_to_stable_ar(rho)
    assert ok
    assert is_stable_ar(rho2)


# -------------------------------------------------------------------
# fitting
# -------------------------------------------------------------------


@pytest.mark.parametrize("method", ["least_squares", "yule_walker"])
@pytest.mark.parametrize(
    "rho_true,sigma_true",
    [
        (np.array([0.6]), 0.4),  # AR(1)
        (np.array([0.5, -0.2, 0.1]), 0.3),  # AR(3)
    ],
)
def test_fit_arp_recovery(method, rho_true, sigma_true):
    x = simulate_arp(rho_true, sigma_true)

    p = len(rho_true)
    rho_hat, sigma_hat = fit_ar_p_pooled(x, p=p, method=method)

    assert rho_hat.shape == (p,)
    assert np.allclose(rho_hat, rho_true, atol=0.05)
    assert abs(sigma_hat - sigma_true) < 0.05


def test_fit_ar_p_ls_errors_if_T_le_p():
    x = np.random.randn(3, 2)
    with pytest.raises(ValueError):
        fit_ar_p_ls_pooled(x, p=3)


def test_fit_ar_p_invalid_method():
    x = np.random.randn(50, 2)
    with pytest.raises(ValueError):
        fit_ar_p_pooled(x, p=1, method="invalid")


# -------------------------------------------------------------------
# wrappers
# -------------------------------------------------------------------


def test_fit_baseline_poly_ar1_recovery():
    T, K = 3000, 3
    phi_true = 0.6
    sigma_true = 0.4

    x = np.random.randn(T, K)
    coefs = np.array([0.0, 1.0])  # identity mean model
    noise = simulate_arp([phi_true], sigma_true, T=T, K=K)
    u = x + noise

    phi_hat, sigma_hat = fit_baseline_poly_ar(x, u, coefs, p=1)

    assert abs(phi_hat - phi_true) < 0.05
    assert abs(sigma_hat - sigma_true) < 0.05


def test_fit_baseline_poly_ar_ar2():
    T, K = 3000, 2
    rho_true = np.array([0.5, -0.2])
    sigma_true = 0.3

    x = np.random.randn(T, K)
    coefs = np.array([0.0, 1.0])
    noise = simulate_arp(rho_true, sigma_true, T=T, K=K)
    u = x + noise

    rho_hat, sigma_hat = fit_baseline_poly_ar(x, u, coefs, p=2)

    assert rho_hat.shape == (2,)
    assert np.allclose(rho_hat, rho_true, atol=0.05)
    assert abs(sigma_hat - sigma_true) < 0.05


def test_fit_flow_latent_ar1_recovery_with_burnin():
    phi_true = 0.5
    sigma_true = 0.3

    x = simulate_arp([phi_true], sigma_true, T=3000)
    phi_hat, sigma_hat = fit_flow_latent_ar(x, p=1, burn_in=50)

    assert abs(phi_hat - phi_true) < 0.05
    assert abs(sigma_hat - sigma_true) < 0.05


def test_fit_flow_latent_ar_ar3_with_burnin():
    rho_true = np.array([0.4, -0.1, 0.05])
    sigma_true = 0.25

    x = simulate_arp(rho_true, sigma_true, T=3000)
    rho_hat, sigma_hat = fit_flow_latent_ar(x, p=3, burn_in=50)

    assert rho_hat.shape == (3,)
    assert np.allclose(rho_hat, rho_true, atol=0.05)
    assert abs(sigma_hat - sigma_true) < 0.05


def test_fit_flow_latent_ar_burnin_too_large():
    x = np.random.randn(10, 2)
    with pytest.raises(ValueError):
        fit_flow_latent_ar(x, p=3, burn_in=9)
