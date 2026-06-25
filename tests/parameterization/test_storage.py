import numpy as np

from parameterization.utils.storage import (
    load_ar_p_parameters,
    load_poly_coefficients,
    save_ar_p_parameters,
    save_poly_coefficients,
)


def test_save_load_poly_coefficients(temp_dir, poly_coefs):
    save_poly_coefficients(str(temp_dir), poly_coefs)
    loaded = load_poly_coefficients(str(temp_dir))

    assert loaded.shape == poly_coefs.shape
    np.testing.assert_allclose(loaded, poly_coefs)


def test_save_load_ar_p_parameters_p1_scalar(temp_dir, ar1_params):
    rho, sigma = ar1_params
    save_ar_p_parameters(str(temp_dir), rho, sigma, ar_order=1)
    loaded_rho, loaded_sigma = load_ar_p_parameters(str(temp_dir), ar_order=1)

    assert np.isscalar(loaded_rho)
    assert np.isscalar(loaded_sigma)
    np.testing.assert_allclose(loaded_rho, rho)
    np.testing.assert_allclose(loaded_sigma, sigma)


def test_save_load_ar_p_parameters_p2_vector(temp_dir):
    rho = np.array([0.73, 0.66], dtype=float)
    sigma = 0.42
    ar_order = 2

    save_ar_p_parameters(str(temp_dir), rho, sigma, ar_order)
    loaded_rho, loaded_sigma = load_ar_p_parameters(str(temp_dir), ar_order)

    assert isinstance(loaded_rho, np.ndarray)
    assert loaded_rho.shape == rho.shape
    np.testing.assert_allclose(loaded_rho, rho)
    assert np.isclose(loaded_sigma, sigma)
