import numpy as np
from pytest import mark

from models.forcing_schedule import (
    ConstantForcingSchedule,
    LinearForcingSchedule,
    OscillatingForcingSchedule,
    forcing_at_array,
)
from parameterization.utils.helpers import (
    compute_ar_p_noise,
    compute_coupling_from_x,
)


def test_compute_ar_p_noise_shape(ar1_params, K):
    phi, sigma_e = ar1_params
    steps = 10
    noise = compute_ar_p_noise(phi, sigma_e, steps, k=K, seed=7)
    assert noise.shape == (steps, K)
    assert np.all(np.isfinite(noise))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_compute_coupling_from_x_shape(K, f_schedule):
    x = np.random.randn(100, K)
    dt = 0.1
    h, b, c = 1.0, 10.0, 10.0

    t_vals = np.arange(x.shape[0]) * dt
    F_values = forcing_at_array(f_schedule, t_vals)

    u_train, x_train = compute_coupling_from_x(x, dt, F_values, h, b, c)

    assert u_train.shape == (99, K)
    assert x_train.shape == (99, K)
    assert np.array_equal(x_train, x[:-1])
    assert np.all(np.isfinite(u_train))
    assert np.all(np.isfinite(x_train))


@mark.parametrize(
    "f_schedule",
    [
        ConstantForcingSchedule(20.0),
        LinearForcingSchedule(F0=18.0, F1=22.0, t0=0.0, t1=5.0),
        OscillatingForcingSchedule(Fmean=20.0, amp=2.0, freq=0.5),
    ],
)
def test_compute_coupling_from_x_scaling(K, f_schedule):
    x = np.random.randn(30, K)
    dt = 0.1
    h, b, c = 1.0, 10.0, 10.0

    t_vals = np.arange(x.shape[0]) * dt
    F_values = forcing_at_array(f_schedule, t_vals)

    u_train_1, _ = compute_coupling_from_x(x, dt, F_values, h, b, c)
    u_train_2, _ = compute_coupling_from_x(x, dt, F_values, h * 2, b, c)

    np.testing.assert_allclose(u_train_2, u_train_1 * 2, rtol=1e-10)
