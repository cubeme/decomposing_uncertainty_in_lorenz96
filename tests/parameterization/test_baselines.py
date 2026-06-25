import numpy as np
from parameterization.baselines.polynomial_ar_p_parameterization import (
    PolynomialARpParameterization,
    load_polynomial_ar_p_parameterization,
)

from parameterization.baselines.fit_parameters import (
    fit_deterministic_poly_coefs,
)
from parameterization.baselines.polynomial_parameterization import (
    PolynomialParameterization,
    load_polynomial_parameterization,
)


######################## Parameter fitting ################################
def test_fit_deterministic_poly_parameterization(sample_x, sample_u, poly_order, K):
    coefs = fit_deterministic_poly_coefs(sample_x, sample_u, poly_order)
    assert coefs.shape == (poly_order + 1,)
    assert np.all(np.isfinite(coefs))


######################## PolynomialParameterization ########################


def test_polynomial_parameterization_predict(sample_x, poly_coefs):
    param = PolynomialParameterization(poly_coefs)
    out = param.predict(sample_x)

    assert out.shape == sample_x.shape
    assert np.all(np.isfinite(out))


def test_polynomial_parameterization_update(sample_x, poly_coefs):
    param = PolynomialParameterization(poly_coefs)
    out_before = param.predict(sample_x)
    param.update()
    out_after = param.predict(sample_x)

    # For deterministic parameterization, update should not change output
    assert np.array_equal(out_before, out_after)


def test_polynomial_parameterization_save_load(sample_x, poly_coefs, tmp_path):
    # Create and save parameterization
    param = PolynomialParameterization(poly_coefs)

    save_file = tmp_path / "param.p"
    param.save(save_file)

    # Load parameterization
    loaded_param = load_polynomial_parameterization(save_file)
    # Verify loaded parameterization has same coefficients
    assert np.array_equal(loaded_param.coefs, param.coefs)

    # Verify it produces same output
    out_original = param.predict(sample_x)
    out_loaded = loaded_param.predict(sample_x)
    assert np.array_equal(out_original, out_loaded)


######################## PolynomialARpParameterization #####################


def test_polynomial_ar_p_parameterization_predict(sample_x, poly_coefs, ar1_params):
    rho, sigma = ar1_params
    param = PolynomialARpParameterization(poly_coefs, rho, sigma, seed=5)
    out = param.predict(sample_x)

    assert out.shape == sample_x.shape
    assert np.all(np.isfinite(out))


def test_polynomial_ar_p_parameterization_update(sample_x, poly_coefs, ar1_params):
    rho, sigma = ar1_params
    param = PolynomialARpParameterization(poly_coefs, rho, sigma, seed=5)
    noise_before = param.noise.copy()
    out_before = param.predict(sample_x)

    param.update()

    noise_after = param.noise.copy()
    out_after = param.predict(sample_x)

    # For AR1 parameterization, update should change the noise state
    assert not np.array_equal(noise_before, noise_after)
    # Output should also change (unless rho and sigma are all zeros)
    if not (np.all(rho == 0) and np.all(sigma == 0)):
        assert not np.array_equal(out_before, out_after)


def test_polynomial_ar_p_parameterization_save_load(
    sample_x, poly_coefs, ar1_params, tmp_path
):
    # Create and save parameterization
    rho, sigma = ar1_params
    param = PolynomialARpParameterization(poly_coefs, rho, sigma, seed=5)

    # Get initial prediction before saving
    out_before_save = param.predict(sample_x)
    noise_before_save = param.noise.copy()

    save_file = tmp_path / "param_ar_p.p"
    param.save(save_file)

    # Load parameterization
    loaded_param = load_polynomial_ar_p_parameterization(save_file)

    # Verify loaded parameterization has same parameters
    assert np.array_equal(loaded_param.coefs, param.coefs)
    assert np.array_equal(loaded_param.rho, param.rho)
    assert np.array_equal(loaded_param.sigma, param.sigma)
    assert loaded_param.seed == param.seed

    # Verify noise state is preserved
    assert np.array_equal(loaded_param.noise, noise_before_save)

    # Verify it produces same output
    out_loaded = loaded_param.predict(sample_x)
    assert np.array_equal(out_before_save, out_loaded)


def test_polynomial_ar_p_parameterization_p2_update_and_shape(sample_x, poly_coefs):
    rho = np.array([0.5, -0.2], dtype=float)
    sigma = 0.07
    param = PolynomialARpParameterization(poly_coefs, rho, sigma, seed=5)

    out_before = param.predict(sample_x)
    param.update()
    out_after = param.predict(sample_x)

    assert out_before.shape == sample_x.shape
    assert out_after.shape == sample_x.shape
    assert np.all(np.isfinite(out_after))
    assert len(param._noise_hist) == 2
    assert not np.array_equal(out_before, out_after)
