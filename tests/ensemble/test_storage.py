import numpy as np
import pytest
import zarr

from ensemble.storage import (
    load_output_gcm_ensemble,
    load_output_l96_ensemble,
    merge_output_l96_ensemble,
    save_output_gcm_ensemble,
    save_output_l96_ensemble,
    save_output_l96_ensemble_split,
)


def test_save_load_gcm_ensemble_numpy(temp_dir, sample_gcm_ensemble_data):
    """Test saving and loading GCM ensemble with numpy backend."""
    x, t = sample_gcm_ensemble_data

    save_output_gcm_ensemble(temp_dir, x, t, backend="numpy")

    x_loaded, t_loaded = load_output_gcm_ensemble(temp_dir, backend="numpy")

    assert x_loaded.shape == x.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(t_loaded, t)


def test_save_load_gcm_ensemble_zarr(
    temp_dir, sample_gcm_ensemble_data, gcm_ensemble_config
):
    """Test saving and loading GCM ensemble with zarr backend."""
    x, t = sample_gcm_ensemble_data

    save_output_gcm_ensemble(temp_dir, x, t, backend="zarr")

    x_loaded, t_loaded = load_output_gcm_ensemble(temp_dir, backend="zarr")

    assert x_loaded.shape == x.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded[:], x)
    assert np.allclose(t_loaded[:], t)


def test_save_load_l96_ensemble_numpy(temp_dir, sample_l96_ensemble_data):
    """Test saving and loading L96 ensemble with numpy backend."""
    x, y, t = sample_l96_ensemble_data

    save_output_l96_ensemble(temp_dir, x, y, t, backend="numpy")

    x_loaded, y_loaded, t_loaded = load_output_l96_ensemble(
        temp_dir, load_y=True, backend="numpy"
    )

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded, t)


def test_save_load_l96_ensemble_zarr(temp_dir, sample_l96_ensemble_data):
    """Test saving and loading L96 ensemble with zarr backend."""
    x, y, t = sample_l96_ensemble_data

    save_output_l96_ensemble(temp_dir, x, y, t, backend="zarr")

    x_loaded, y_loaded, t_loaded = load_output_l96_ensemble(
        temp_dir, load_y=True, backend="zarr"
    )

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded[:], x)
    assert np.allclose(y_loaded[:], y)
    assert np.allclose(t_loaded[:], t)


def test_load_l96_ensemble_without_y(temp_dir, sample_l96_ensemble_data):
    """Test loading L96 ensemble without y variables."""
    x, y, t = sample_l96_ensemble_data

    save_output_l96_ensemble(temp_dir, x, y, t, backend="numpy")

    x_loaded, t_loaded = load_output_l96_ensemble(
        temp_dir, load_y=False, backend="numpy"
    )

    assert x_loaded.shape == x.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(t_loaded, t)


def test_save_l96_ensemble_split(temp_dir, sample_l96_ensemble_data):
    """Test saving L96 ensemble in split format."""
    x, y, t = sample_l96_ensemble_data

    save_output_l96_ensemble_split(temp_dir, x, y, t, split=0, backend="numpy")
    save_output_l96_ensemble_split(temp_dir, x, y, t, split=1, backend="numpy")

    # Check files exist
    assert (temp_dir / "x_0.npy").exists()
    assert (temp_dir / "y_0.npy").exists()
    assert (temp_dir / "t_0.npy").exists()
    assert (temp_dir / "x_1.npy").exists()

    # Load and verify
    x_loaded = np.load(temp_dir / "x_0.npy")
    assert x_loaded.shape == x.shape
    assert np.allclose(x_loaded, x)


def test_save_l96_ensemble_split_zarr(temp_dir, sample_l96_ensemble_data):
    """Test saving L96 ensemble in split format with zarr backend."""
    x, y, t = sample_l96_ensemble_data

    save_output_l96_ensemble_split(temp_dir, x, y, t, split=0, backend="zarr")
    save_output_l96_ensemble_split(temp_dir, x, y, t, split=1, backend="zarr")

    assert (temp_dir / "x_0.zarr").exists()
    assert (temp_dir / "y_0.zarr").exists()
    assert (temp_dir / "t_0.zarr").exists()
    assert (temp_dir / "x_1.zarr").exists()

    # Load and verify
    x_loaded = zarr.open_array(temp_dir / "x_0.zarr", mode="r")
    assert x_loaded.shape == x.shape
    assert np.allclose(x_loaded[:], x)


def test_merge_l96_ensemble_numpy(temp_dir, sample_l96_ensemble_data):
    """Test merging L96 ensemble splits with numpy backend."""
    x, y, t = sample_l96_ensemble_data

    # Save in split format
    save_output_l96_ensemble_split(temp_dir, x[:1], y[:1], t, split=0, backend="numpy")
    save_output_l96_ensemble_split(temp_dir, x[1:], y[1:], t, split=1, backend="numpy")

    # Merge
    merge_output_l96_ensemble(temp_dir, merge_y=True, backend="numpy")

    # Load merged
    x_loaded, y_loaded, t_loaded = load_output_l96_ensemble(
        temp_dir, load_y=True, backend="numpy"
    )

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded, x)
    assert np.allclose(y_loaded, y)
    assert np.allclose(t_loaded, t)


def test_merge_l96_ensemble_zarr(temp_dir, sample_l96_ensemble_data):
    """Test merging L96 ensemble splits with zarr backend."""
    x, y, t = sample_l96_ensemble_data

    # Save in split format
    save_output_l96_ensemble_split(temp_dir, x[:1], y[:1], t, split=0, backend="zarr")
    save_output_l96_ensemble_split(temp_dir, x[1:], y[1:], t, split=1, backend="zarr")

    # Merge
    merge_output_l96_ensemble(temp_dir, merge_y=True, backend="zarr")

    # Load merged
    x_loaded, y_loaded, t_loaded = load_output_l96_ensemble(
        temp_dir, load_y=True, backend="zarr"
    )

    assert x_loaded.shape == x.shape
    assert y_loaded.shape == y.shape
    assert t_loaded.shape == t.shape
    assert np.allclose(x_loaded[:], x)
    assert np.allclose(y_loaded[:], y)
    assert np.allclose(t_loaded[:], t)


def test_invalid_backend_save_gcm(temp_dir, sample_gcm_ensemble_data):
    """Test that invalid backend raises ValueError for GCM save."""
    x, t = sample_gcm_ensemble_data

    with pytest.raises(ValueError, match="Unsupported backend"):
        save_output_gcm_ensemble(temp_dir, x, t, backend="invalid")


def test_invalid_backend_load_gcm(temp_dir):
    """Test that invalid backend raises ValueError for GCM load."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        load_output_gcm_ensemble(temp_dir, backend="invalid")


def test_invalid_backend_save_l96(temp_dir, sample_l96_ensemble_data):
    """Test that invalid backend raises ValueError for L96 save."""
    x, y, t = sample_l96_ensemble_data

    with pytest.raises(ValueError, match="Unsupported backend"):
        save_output_l96_ensemble(temp_dir, x, y, t, backend="invalid")


def test_invalid_backend_save_l96_split(temp_dir, sample_l96_ensemble_data):
    """Test that invalid backend raises ValueError for L96 split save."""
    x, y, t = sample_l96_ensemble_data

    with pytest.raises(ValueError, match="Unsupported backend"):
        save_output_l96_ensemble_split(temp_dir, x, y, t, split=0, backend="invalid")


def test_invalid_backend_load_l96(temp_dir):
    """Test that invalid backend raises ValueError for L96 load."""
    with pytest.raises(ValueError, match="Unsupported backend"):
        load_output_l96_ensemble(temp_dir, backend="invalid")
