import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from models.forcing_schedule import ConstantForcingSchedule

@pytest.fixture
def K():
    return 4  # small for fast tests


@pytest.fixture
def poly_order():
    return 3


@pytest.fixture
def sample_x(K):
    np.random.seed(0)
    return np.random.randn(20, K)


@pytest.fixture
def sample_u(K):
    np.random.seed(1)
    return np.random.randn(20, K)


@pytest.fixture
def poly_coefs(K, poly_order):
    np.random.seed(2)
    # More realistic: coefficients typically O(0.01) to O(1)
    return np.random.randn(poly_order + 1) * 0.1


@pytest.fixture
def ar1_params(K):
    np.random.seed(3)
    phi = np.random.uniform(0.2, 0.8)
    sigma_e = np.random.uniform(0.01, 0.1)
    return phi, sigma_e


@pytest.fixture
def dummy_config(poly_order):
    # Minimal config namespace with attributes used in helpers
    return SimpleNamespace(
        poly_order=poly_order,
        si=0.1,
        f_schedule=ConstantForcingSchedule(20.0),
        h=1.0,
        b=10.0,
        c=4.0,
        seed=42,
    )


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)
