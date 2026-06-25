import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from models.forcing_schedule import (
    ConstantForcingSchedule,
)
from parameterization.baselines.polynomial_ar_p_parameterization import (
    PolynomialARpParameterization,
)
from parameterization.baselines.polynomial_parameterization import (
    PolynomialParameterization,
)


@pytest.fixture
def l96_config():
    """Fixture providing default configuration for model tests."""
    return SimpleNamespace(
        K=8,
        J=32,
        f_schedule=ConstantForcingSchedule(20.0),
        h=1.0,
        b=10.0,
        c=10.0,
        seed=42,
        dt=0.005,
        si=0.05,
        y_scale=1.0,
        total_time=5,
        spin_up_time=0,  # Skip spin-up by default
    )


@pytest.fixture
def gcm_config():
    """Fixture providing default configuration for model tests."""
    return SimpleNamespace(
        K=8,
        J=32,
        f_schedule=ConstantForcingSchedule(20.0),
        dt=0.01,
        si=0.05,
        total_time=5,
        time_stepping="RK2",
        seed=42,
    )


@pytest.fixture
def sample_x(l96_config):
    """Sample slow variables (X) from Lorenz 96."""
    np.random.seed(0)
    return np.random.randn(l96_config.K)


@pytest.fixture
def sample_y(l96_config):
    """Sample fast variables (Y) from Lorenz 96."""
    np.random.seed(1)
    return np.random.randn(l96_config.K * l96_config.J)


@pytest.fixture
def coefs():
    """Polynomial coefficients for parameterization. Arnold et al. 2013"""
    return np.array([0.341, 1.3, -0.0136, -0.00235])


@pytest.fixture
def poly_order(coefs):
    """Polynomial order for parameterization tests."""
    return len(coefs) - 1


@pytest.fixture
def polynomial_parameterization(l96_config, coefs):
    """Zero polynomial parameterization."""
    return PolynomialParameterization(coefs)


@pytest.fixture
def polynomial_ar1_parameterization(l96_config, coefs):
    """Polynomial AR1 parameterization."""
    np.random.seed(2)
    rho = float(np.random.uniform(0.2, 0.8))
    sigma = float(np.random.uniform(0.01, 0.1))
    return PolynomialARpParameterization(coefs, rho, sigma, seed=l96_config.seed)


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
