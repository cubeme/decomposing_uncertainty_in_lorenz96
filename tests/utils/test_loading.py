""

import json
from pathlib import Path

import numpy as np

from utils.loading import load_flags, load_initial_states, load_seeds, load_sweep
from utils.saving import save_initial_states, save_seeds


def test_load_initial_states_shapes(temp_dir, sample_initial_states):
    """Test that load_initial_states returns correct shapes."""
    x, y, t = sample_initial_states

    # Save first
    save_initial_states(temp_dir, x, y, t)

    # Load
    x_loaded, y_loaded, t_loaded = load_initial_states(temp_dir)

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape


def test_load_initial_states_values(temp_dir, sample_initial_states):
    """Test that load_initial_states returns correct values."""
    x, y, t = sample_initial_states

    save_initial_states(temp_dir, x, y, t)
    x_loaded, y_loaded, t_loaded = load_initial_states(temp_dir)

    assert np.allclose(x_loaded, x)
    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded, t)


def test_load_seeds_shape(temp_dir, sample_seeds):
    """Test that load_seeds returns correct shape."""
    save_seeds(temp_dir, sample_seeds)

    seeds_loaded = load_seeds(temp_dir)

    assert seeds_loaded.shape == sample_seeds.shape


def test_load_seeds_values(temp_dir, sample_seeds):
    """Test that load_seeds returns correct values."""
    save_seeds(temp_dir, sample_seeds)

    seeds_loaded = load_seeds(temp_dir)

    assert np.array_equal(seeds_loaded, sample_seeds)


def test_load_sweep(temp_dir):
    """Test that load_sweep loads JSON correctly."""
    sweep_data = {
        "param1": [1, 2, 3],
        "param2": [0.1, 0.2, 0.3],
        "param3": ["a", "b", "c"],
    }

    # Save sweep file
    with open(temp_dir / "sweep.json", "w") as fp:
        json.dump(sweep_data, fp)

    # Load
    loaded_sweep = load_sweep(temp_dir)

    assert loaded_sweep == sweep_data
    assert "param1" in loaded_sweep
    assert loaded_sweep["param1"] == [1, 2, 3]


def test_load_flags(temp_dir):
    """Test that load_flags loads JSON correctly."""
    flags_data = {
        "experiment_name": "test_experiment",
        "K": 8,
        "f_schedule": {"type": "constant", "F": 20.0},
        "n_ens_members": 10,
        "flag_bool": True,
    }

    # Save flags file
    with open(temp_dir / "flags.json", "w") as fp:
        json.dump(flags_data, fp)

    # Load
    loaded_flags = load_flags(temp_dir)

    assert loaded_flags == flags_data
    assert loaded_flags["experiment_name"] == "test_experiment"
    assert loaded_flags["K"] == 8
    assert loaded_flags["f_schedule"]["F"] == 20.0


def test_load_flags_has_correct_types(temp_dir):
    """Test that loaded flags preserve correct types."""
    flags_data = {
        "int_value": 42,
        "float_value": 3.14,
        "str_value": "test",
        "bool_value": True,
        "list_value": [1, 2, 3],
    }

    with open(temp_dir / "flags.json", "w") as fp:
        json.dump(flags_data, fp)

    loaded = load_flags(temp_dir)

    assert isinstance(loaded["int_value"], int)
    assert isinstance(loaded["float_value"], float)
    assert isinstance(loaded["str_value"], str)
    assert isinstance(loaded["bool_value"], bool)
    assert isinstance(loaded["list_value"], list)


def test_load_functions_work_with_path_objects(
    temp_dir, sample_initial_states, sample_seeds
):
    """Test that all load functions accept Path objects."""

    x, y, t = sample_initial_states
    save_initial_states(temp_dir, x, y, t)
    save_seeds(temp_dir, sample_seeds)

    # Test with Path object
    path_obj = Path(temp_dir)

    x_loaded, y_loaded, t_loaded = load_initial_states(path_obj)
    assert x_loaded.shape == x.shape

    seeds_loaded = load_seeds(path_obj)
    assert seeds_loaded.shape == sample_seeds.shape


def test_load_functions_work_with_string_paths(temp_dir, sample_initial_states):
    """Test that all load functions accept string paths."""
    x, y, t = sample_initial_states
    save_initial_states(temp_dir, x, y, t)

    # Test with string path
    str_path = str(temp_dir)

    x_loaded, y_loaded, t_loaded = load_initial_states(str_path)
    assert x_loaded.shape == x.shape
