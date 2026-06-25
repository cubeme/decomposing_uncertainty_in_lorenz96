from pathlib import Path

import numpy as np

from utils.saving import save_initial_states, save_seeds


def test_save_initial_states_creates_files(temp_dir, sample_initial_states):
    """Test that save_initial_states creates all necessary files."""
    x, y, t = sample_initial_states

    save_initial_states(temp_dir, x, y, t)

    init_states_dir = temp_dir / "initial_states"
    assert (init_states_dir / "x.npy").exists()
    assert (init_states_dir / "y.npy").exists()
    assert (init_states_dir / "t.npy").exists()


def test_save_initial_states_correct_shapes(temp_dir, sample_initial_states):
    """Test that saved initial states have correct shapes when loaded."""
    x, y, t = sample_initial_states

    save_initial_states(temp_dir, x, y, t)

    init_states_dir = temp_dir / "initial_states"
    x_loaded = np.load(init_states_dir / "x.npy")
    y_loaded = np.load(init_states_dir / "y.npy")
    t_loaded = np.load(init_states_dir / "t.npy")

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape


def test_save_initial_states_correct_values(temp_dir, sample_initial_states):
    """Test that saved initial states have correct values."""
    x, y, t = sample_initial_states

    save_initial_states(temp_dir, x, y, t)

    init_states_dir = temp_dir / "initial_states"
    x_loaded = np.load(init_states_dir / "x.npy")
    y_loaded = np.load(init_states_dir / "y.npy")
    t_loaded = np.load(init_states_dir / "t.npy")

    assert np.allclose(x_loaded, x)
    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded, t)


def test_save_seeds_creates_file(temp_dir, sample_seeds):
    """Test that save_seeds creates file."""
    save_seeds(temp_dir, sample_seeds)

    seeds_dir = temp_dir / "seeds"
    assert (seeds_dir / "seeds.npy").exists()


def test_save_seeds_correct_shape(temp_dir, sample_seeds):
    """Test that saved seeds have correct shape."""
    save_seeds(temp_dir, sample_seeds)

    seeds_dir = temp_dir / "seeds"
    seeds_loaded = np.load(seeds_dir / "seeds.npy")

    assert seeds_loaded.shape == sample_seeds.shape


def test_save_seeds_correct_values(temp_dir, sample_seeds):
    """Test that saved seeds have correct values."""
    save_seeds(temp_dir, sample_seeds)

    seeds_dir = temp_dir / "seeds"
    seeds_loaded = np.load(seeds_dir / "seeds.npy")

    assert np.array_equal(seeds_loaded, sample_seeds)


def test_save_functions_work_with_path_objects(
    temp_dir, sample_initial_states, sample_seeds
):
    """Test that save functions accept Path objects."""

    x, y, t = sample_initial_states
    path_obj = Path(temp_dir)

    # Should not raise error
    save_initial_states(path_obj, x, y, t)
    save_seeds(path_obj, sample_seeds)

    # Verify files exist
    assert (temp_dir / "initial_states" / "x.npy").exists()
    assert (temp_dir / "seeds" / "seeds.npy").exists()


def test_save_functions_work_with_string_paths(
    temp_dir, sample_initial_states, sample_seeds
):
    """Test that save functions accept string paths."""
    x, y, t = sample_initial_states
    str_path = str(temp_dir)

    # Should not raise error
    save_initial_states(str_path, x, y, t)
    save_seeds(str_path, sample_seeds)

    # Verify files exist
    assert (temp_dir / "initial_states" / "x.npy").exists()
    assert (temp_dir / "seeds" / "seeds.npy").exists()
