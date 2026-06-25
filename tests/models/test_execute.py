from types import SimpleNamespace

import numpy as np
from pytest import mark

from models.execute import (
    initialize_l96,
    initialize_poly_ar_p_param_gcm,
    initialize_poly_param_gcm,
    run_gcm,
    run_l96,
)
from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
)


def test_initialize_poly_param_gcm_default(gcm_config: SimpleNamespace):
    """Test initialize_poly_param_gcm with default zero parameterization."""
    gcm = initialize_poly_param_gcm(gcm_config)

    assert gcm is not None
    assert gcm.f_schedule is gcm_config.f_schedule
    assert gcm.parameterization is not None


def test_initialize_poly_param_gcm_with_coefs(gcm_config: SimpleNamespace, coefs):
    """Test initialize_poly_param_gcm with provided coefficients."""
    gcm = initialize_poly_param_gcm(gcm_config, coefs=coefs)

    assert gcm is not None
    assert np.allclose(gcm.parameterization.coefs, coefs)


# todo: test for p=2 as well. use parameterized test with different coefs, rho, sigma, ar_order values. test that rho and sigma are set correctly in the parameterization. test that the AR noise process is generated correctly (e.g. by checking the autocorrelation of the noise).
@mark.parametrize(
    "rho,sigma,ar_order",
    [
        (0.65, 0.06, 1),
        (np.array([0.45, -0.15]), 0.08, 2),
        (np.array([0.3, 0.1]), 0.04, 2),
    ],
)
def test_initialize_poly_ar_p_param_gcm(
    gcm_config: SimpleNamespace, coefs, rho, sigma, ar_order
):
    """Test initialize_poly_ar_p_param_gcm for AR(1)/AR(2) parameterizations."""
    gcm = initialize_poly_ar_p_param_gcm(gcm_config, coefs, rho, sigma)
    param = gcm.parameterization
    expected_rho = np.asarray(rho, dtype=float).reshape(-1)

    assert gcm is not None
    assert np.allclose(param.coefs, coefs)
    assert param.rho.shape == (ar_order,)
    assert np.allclose(param.rho, expected_rho)
    assert np.isclose(param.sigma, sigma)

    # Validate AR(p) stochastic noise via sample autocorrelation.
    burn_in = 200
    n_samples = 2500
    noise = np.empty(n_samples)
    for i in range(burn_in + n_samples):
        param.update()
        if i >= burn_in:
            noise[i - burn_in] = float(param.noise[0])

    centered = noise - noise.mean()
    var = float(np.dot(centered, centered))
    assert var > 0.0

    def sample_autocorr(lag: int) -> float:
        return float(np.dot(centered[:-lag], centered[lag:]) / var)

    if ar_order == 1:
        assert np.isclose(sample_autocorr(1), expected_rho[0], atol=0.08)
    else:
        rho1, rho2 = expected_rho
        expected_acf_lag1 = rho1 / (1.0 - rho2)
        expected_acf_lag2 = rho1 * expected_acf_lag1 + rho2
        assert np.isclose(sample_autocorr(1), expected_acf_lag1, atol=0.10)
        assert np.isclose(sample_autocorr(2), expected_acf_lag2, atol=0.12)


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_run_gcm_shapes(gcm_config: SimpleNamespace, f_schedule, coefs):
    """Test that run_gcm returns correct shapes."""
    gcm_config.f_schedule = f_schedule
    gcm = initialize_poly_param_gcm(gcm_config, coefs=coefs)

    init_conditions = np.random.randn(gcm_config.K)

    hist, time = run_gcm(gcm, init_conditions, gcm_config)

    nt = int(gcm_config.total_time / gcm_config.si)

    assert hist.shape == (nt + 1, gcm_config.K)
    assert time.shape == (nt + 1,)
    assert np.all(np.isfinite(hist))
    assert np.all(np.isfinite(time))


def test_run_gcm_time_correct(gcm_config: SimpleNamespace, coefs):
    """Test that run_gcm produces correct time array."""
    gcm = initialize_poly_param_gcm(gcm_config, coefs=coefs)
    init_conditions = np.random.randn(gcm_config.K)
    hist, time = run_gcm(gcm, init_conditions, gcm_config)

    # Should start at 0 and reach total_time
    assert np.isclose(time[0], 0.0)
    assert np.isclose(time[-1], gcm_config.total_time, rtol=1e-10)


def test_initialize_l96_basic(l96_config):
    """Test basic L96 initialization."""
    l96 = initialize_l96(l96_config)

    assert l96 is not None
    assert l96.k == l96_config.K
    assert l96.j == l96_config.J
    assert l96.f_schedule is l96_config.f_schedule


def test_initialize_l96_with_custom_seed(l96_config):
    """Test L96 initialization with custom seed."""
    l96_a = initialize_l96(l96_config, seed=42)
    l96_b = initialize_l96(l96_config, seed=42)

    # Same seed should give same initial conditions
    assert np.allclose(l96_a.x, l96_b.x)
    assert np.allclose(l96_a.y, l96_b.y)


def test_initialize_l96_y_scale_changes_fast_variable_magnitude(l96_config):
    """Test that y_scale changes the initialized fast variables."""
    config_default = SimpleNamespace(**vars(l96_config))
    config_scaled = SimpleNamespace(**vars(l96_config))
    config_default.y_scale = 1.0
    config_scaled.y_scale = 2.5

    l96_default = initialize_l96(config_default, seed=42)
    l96_scaled = initialize_l96(config_scaled, seed=42)

    assert np.allclose(l96_default.x, l96_scaled.x)
    assert np.allclose(l96_scaled.y, 2.5 * l96_default.y)


def test_initialize_l96_no_spinup(l96_config):
    """Test L96 initialization without spin-up."""
    l96_config.spin_up_time = 0

    l96 = initialize_l96(l96_config)

    assert l96 is not None


def test_initialize_l96_with_spinup(l96_config):
    """Test L96 initialization with spin-up."""
    config_spinup = l96_config
    config_spinup.spin_up_time = 5

    l96 = initialize_l96(config_spinup)

    # After spin-up, state should have evolved
    assert l96.t == 0
    assert np.all(np.isfinite(l96.x))
    assert np.all(np.isfinite(l96.y))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_run_l96_shapes(l96_config, f_schedule):
    """Test that run_l96 returns correct shapes."""
    l96_config.f_schedule = f_schedule
    l96 = initialize_l96(l96_config)

    x_hist, y_hist, time = run_l96(l96, l96_config, store=False)

    nt = int(l96_config.total_time / l96_config.si)

    assert x_hist.shape == (nt + 1, l96_config.K)
    assert y_hist.shape == (nt + 1, l96_config.K * l96_config.J)
    assert time.shape == (nt + 1,)

    assert np.all(np.isfinite(x_hist))
    assert np.all(np.isfinite(y_hist))
    assert np.all(np.isfinite(time))


def test_run_l96_time_correct(l96_config):
    """Test that run_l96 produces correct time array."""
    l96 = initialize_l96(l96_config)

    x_hist, y_hist, time = run_l96(l96, l96_config, store=False)

    # Should reach approximately total_time
    assert np.isclose(time[-1], l96_config.total_time, rtol=1e-10)


def test_run_l96_store_true(l96_config):
    """Test that run_l96 with store=True updates model state."""
    l96 = initialize_l96(l96_config)

    x_hist, y_hist, time = run_l96(l96, l96_config, store=True)

    # Final state should be stored
    assert np.allclose(l96.x, x_hist[-1])
    assert np.allclose(l96.y, y_hist[-1])


def test_run_l96_store_false(l96_config):
    """Test that run_l96 with store=False doesn't update model state."""
    l96 = initialize_l96(l96_config)

    x_original = l96.x.copy()
    y_original = l96.y.copy()

    x_hist, y_hist, time = run_l96(l96, l96_config, store=False)

    # State should remain unchanged
    assert np.allclose(l96.x, x_original)
    assert np.allclose(l96.y, y_original)
