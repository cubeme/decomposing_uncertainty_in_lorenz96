from types import SimpleNamespace

import numpy as np

from models.storage import (
    load_output_gcm,
    load_output_l96,
    save_output_gcm,
    save_output_l96,
)


def test_save_load_output_l96_numpy(temp_dir, l96_config: SimpleNamespace):
    """Test save and load for L96 output."""
    x = np.random.randn(100, l96_config.K)
    y = np.random.randn(100, l96_config.K * l96_config.J)
    t = np.linspace(0, 10, 100)

    # Save
    save_output_l96(temp_dir, x, y, t, backend="numpy")

    # Load
    x_loaded, y_loaded, t_loaded = load_output_l96(temp_dir, backend="numpy")

    # Check shapes and values
    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded, t)


def test_save_load_output_l96_zarr(temp_dir, l96_config: SimpleNamespace):
    """Test save and load for L96 output with zarr backend."""
    x = np.random.randn(100, l96_config.K)
    y = np.random.randn(100, l96_config.K * l96_config.J)
    t = np.linspace(0, 10, 100)

    save_output_l96(temp_dir, x, y, t, backend="zarr")

    x_loaded, y_loaded, t_loaded = load_output_l96(temp_dir, backend="zarr")

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded[:], x)
    assert np.allclose(y_loaded[:], y)
    assert np.allclose(t_loaded[:], t)


def test_save_output_l96_exists_numpy(temp_dir, l96_config: SimpleNamespace):
    """Test that save_output_l96 saves to files."""
    x = np.random.randn(50, l96_config.K)
    y = np.random.randn(50, l96_config.K * l96_config.J)
    t = np.linspace(0, 5, 50)

    # Should not raise
    save_output_l96(temp_dir, x, y, t)

    # Verify files exist
    x_file = temp_dir / "x.npy"
    y_file = temp_dir / "y.npy"
    t_file = temp_dir / "t.npy"

    assert x_file.exists()
    assert y_file.exists()
    assert t_file.exists()


def test_save_output_l96_exists_zarr(temp_dir, l96_config: SimpleNamespace):
    """Test that save_output_l96 saves zarr arrays."""
    x = np.random.randn(50, l96_config.K)
    y = np.random.randn(50, l96_config.K * l96_config.J)
    t = np.linspace(0, 5, 50)

    save_output_l96(temp_dir, x, y, t, backend="zarr")

    assert (temp_dir / "x.zarr").exists()
    assert (temp_dir / "y.zarr").exists()
    assert (temp_dir / "t.zarr").exists()


def test_save_output_l96_invalid_backend(temp_dir, l96_config: SimpleNamespace):
    """Test that invalid backend raises ValueError for L96 save."""
    x = np.random.randn(5, l96_config.K)
    y = np.random.randn(5, l96_config.K * l96_config.J)
    t = np.linspace(0, 1, 5)

    with np.testing.assert_raises(ValueError):
        save_output_l96(temp_dir, x, y, t, backend="invalid")


def test_load_output_l96_invalid_backend(temp_dir):
    """Test that invalid backend raises ValueError for L96 load."""
    with np.testing.assert_raises(ValueError):
        load_output_l96(temp_dir, backend="invalid")


def test_save_load_output_gcm(temp_dir, gcm_config: SimpleNamespace):
    """Test save and load for GCM output."""
    x = np.random.randn(100, gcm_config.K)
    t = np.linspace(0, 10, 100)

    # Save
    save_output_gcm(temp_dir, x, t)

    # Load
    x_loaded, t_loaded = load_output_gcm(temp_dir)

    # Check shapes and values
    assert x_loaded.shape == x.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(t_loaded, t)


def test_save_output_gcm_exists(temp_dir, gcm_config: SimpleNamespace):
    """Test that save_output_gcm saves to files."""
    x = np.random.randn(50, gcm_config.K)
    t = np.linspace(0, 5, 50)

    # Should not raise
    save_output_gcm(temp_dir, x, t)

    # Verify files exist
    x_file = temp_dir / "x.npy"
    t_file = temp_dir / "t.npy"

    assert x_file.exists()
    assert t_file.exists()
