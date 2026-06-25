"""Store and load model simulation outputs."""

from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from utils.saving import save_zarr_array


def save_output_l96(out_path, x, y, t, backend="numpy"):
    """
    Save Lorenz '96 simulation results to the specified output folder.

    Args:
        out_path (str): Path to the output folder.
        x (numpy.ndarray): Array of slow variables (X).
        y (numpy.ndarray): Array of fast variables (Y).
        t (numpy.ndarray): Array of time points.
        backend (str, optional): Storage backend, either 'numpy' or 'zarr'.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if backend == "numpy":
        np.save(out_path / "x.npy", x)
        np.save(out_path / "y.npy", y)
        np.save(out_path / "t.npy", t)
    elif backend == "zarr":
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        save_zarr_array(out_path / "x.zarr", x, compressor)
        save_zarr_array(out_path / "y.zarr", y, compressor)
        save_zarr_array(out_path / "t.zarr", t, compressor)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")


def load_output_l96(load_path, backend="numpy", mmap_mode="r"):
    """
    Load Lorenz '96 simulation results from the specified load path.

    Args:
        load_path (str): Path to the L96 data.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.
        backend (str, optional): Storage backend, either 'numpy' or 'zarr'.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of slow variables (X).
            - y (numpy.ndarray): Array of fast variables (Y).
            - t (numpy.ndarray): Array of time points.
    """
    load_path = Path(load_path)

    if backend == "numpy":
        x = np.load(load_path / "x.npy", mmap_mode=mmap_mode)
        y = np.load(load_path / "y.npy", mmap_mode=mmap_mode)
        t = np.load(load_path / "t.npy", mmap_mode=mmap_mode)
    elif backend == "zarr":
        x = zarr.open_array(load_path / "x.zarr", mode="r")
        y = zarr.open_array(load_path / "y.zarr", mode="r")
        t = zarr.open_array(load_path / "t.zarr", mode="r")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")

    return x, y, t


def save_output_gcm(out_path, x, t):
    """
    Save GCM simulation results to the specified output folder.

    Args:
        out_path (str): Path to the output directory.
        x (numpy.ndarray): Array of state variables.
        t (numpy.ndarray): Array of time points.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "x.npy", x)
    np.save(out_path / "t.npy", t)


def load_output_gcm(load_path):
    """
    Load GCM simulation results from the specified load path.

    Args:
        load_path (str): Path to GCM data.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of state variables.
            - t (numpy.ndarray): Array of time points.
    """
    load_path = Path(load_path)

    x = np.load(load_path / "x.npy")
    t = np.load(load_path / "t.npy")

    return x, t
