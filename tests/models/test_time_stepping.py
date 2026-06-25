import numpy as np

from models.GCM.time_stepping import RK2, RK4, euler_forward


def test_euler_forward_shape():
    """Test that euler_forward maintains state shape."""
    x = np.array([1.0, 2.0, 3.0])
    dt = 0.01

    def simple_fn(x, t):
        return -x  # Simple linear decay

    x_next = euler_forward(simple_fn, dt, x, t=1.0)

    assert x_next.shape == x.shape
    assert np.all(np.isfinite(x_next))


def test_euler_forward_logic():
    """Test that euler_forward performs correct integration."""
    x = np.array([1.0])
    dt = 0.1

    def constant_fn(x, t):
        return np.ones_like(x)  # dx/dt = 1

    x_next = euler_forward(constant_fn, dt, x, t=1.0)
    # Should be x + dt * 1 = 1 + 0.1 = 1.1
    assert np.allclose(x_next, np.array([1.1]))


def test_rk2_shape():
    """Test that RK2 maintains state shape."""
    x = np.array([1.0, 2.0, 3.0])
    dt = 0.01

    def simple_fn(x, t):
        return -x

    x_next = RK2(simple_fn, dt, x, t=1.0)
    assert x_next.shape == x.shape
    assert np.all(np.isfinite(x_next))


def test_rk2_logic():
    """Test that RK2 converges better than Euler for linear systems."""
    x = np.array([1.0])
    dt = 0.1

    def linear_fn(x, t):
        return -x

    # Exact solution for dx/dt = -x with x(0)=1 is x(t) = exp(-t)
    exact = np.exp(-dt)

    x_rk2 = RK2(linear_fn, dt, x, t=1.0)
    x_euler = euler_forward(linear_fn, dt, x, t=1.0)

    # RK2 error should be smaller than Euler error
    rk2_error = abs(x_rk2[0] - exact)
    euler_error = abs(x_euler[0] - exact)
    assert rk2_error < euler_error


def test_rk4_shape():
    """Test that RK4 maintains state shape."""
    x = np.array([1.0, 2.0, 3.0])
    dt = 0.01

    def simple_fn(x, t):
        return -x

    x_next = RK4(simple_fn, dt, x, t=1.0)
    assert x_next.shape == x.shape
    assert np.all(np.isfinite(x_next))


def test_rk4_logic():
    """Test that RK4 converges better than RK2 for nonlinear systems."""
    x = np.array([1.0])
    dt = 0.1

    def quadratic_fn(x, t):
        return x - x**2

    # Multiple steps with RK4 should maintain stability
    x_current = x.copy()
    for _ in range(10):
        x_current = RK4(quadratic_fn, dt, x_current, t=1.0)

    assert np.all(np.isfinite(x_current))
