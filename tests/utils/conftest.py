import tempfile
from pathlib import Path

import numpy as np
import pytest

CONFIGS_DIR = Path(__file__).parent / "configs"


@pytest.fixture
def temp_dir():
    """Temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_initial_states():
    """Sample initial states data."""
    np.random.seed(42)
    n_states = 5
    K = 8
    J = 32
    x = np.random.randn(n_states, K)
    y = np.random.randn(n_states, K * J)
    t = np.linspace(0, 10, n_states)
    return x, y, t


@pytest.fixture
def sample_seeds():
    """Sample random seeds."""
    return np.array([42, 123, 456, 789, 101112])


@pytest.fixture
def configs_dir():
    """Return the directory containing test configs."""
    return CONFIGS_DIR


@pytest.fixture
def output_root(tmp_path):
    """Provide a temporary root for results."""
    return tmp_path / "results"
