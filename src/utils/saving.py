"""Save simulation states and random seeds."""

from pathlib import Path

import numpy as np
import zarr

INIT_STATES_DIR = "initial_states"
SEEDS_DIR = "seeds"


def save_initial_states(out_path, x, y, t):
    out_path = Path(out_path) / INIT_STATES_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "x.npy", x)
    np.save(out_path / "y.npy", y)
    np.save(out_path / "t.npy", t)


def save_seeds(out_path, seeds):
    out_path = Path(out_path) / SEEDS_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    np.save(out_path / "seeds.npy", seeds)


def save_zarr_array(path, array, compressor):
    z = zarr.open(
        path,
        mode="w",
        shape=array.shape,
        dtype=array.dtype,
        compressor=compressor,
        zarr_format=2,
    )
    z[:] = array
