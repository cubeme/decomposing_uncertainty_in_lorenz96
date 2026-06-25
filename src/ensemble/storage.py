"""Store and load ensemble simulation outputs."""

import asyncio
import glob
import os
import re
import shutil
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc
from numpy.lib.format import open_memmap
from zarr.storage import LocalStore

from utils.saving import save_zarr_array


def save_output_gcm_ensemble(out_path, x, t, backend="numpy"):
    """
    Save ensemble simulation results to the specified output path.

    Args:
        out_path (str): Path to the output folder.
        config (ConfigGCM): Configuration object containing settings.
        x (numpy.ndarray): Array of ensemble state variables.
        t (numpy.ndarray): Array of time points.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.
        backend (str, optional): Storage backend, either 'numpy' or 'zarr'. Default is 'numpy'.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if backend == "numpy":
        np.save(out_path / "x.npy", x)
        np.save(out_path / "t.npy", t)

    elif backend == "zarr":
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

        # Save arrays as compressed Zarr files
        save_zarr_array(out_path / "x.zarr", x, compressor)
        save_zarr_array(out_path / "t.zarr", t, compressor)

    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")


def load_output_gcm_ensemble(load_path, backend="numpy", mmap_mode="r"):
    """
    Load ensemble simulation results from the specified load path.

    Args:
        load_path (str): Path to the GCM ensemble data.
        config (ConfigGCM): Configuration object containing settings.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of ensemble state variables.
            - t (numpy.ndarray): Array of time points.
            - init_idx (numpy.ndarray): Array of initial state indices.
    """
    load_path = Path(load_path)
    if backend == "numpy":
        x = np.load(load_path / "x.npy", mmap_mode=mmap_mode)
        t = np.load(load_path / "t.npy", mmap_mode=mmap_mode)
    elif backend == "zarr":
        x = zarr.open_array(load_path / "x.zarr", mode="r")
        t = zarr.open_array(load_path / "t.zarr", mode="r")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")

    return x, t


def save_output_l96_ensemble(
    out_path, x, y, t, save_y=True, backend="numpy", compressor=None
):
    """
    Save ensemble simulation results to the specified output path.

    Args:
        out_path (str): Path to the output folder.
        x (numpy.ndarray): Array of slow variables (X).
        y (numpy.ndarray): Array of fast variables (Y).
        t (numpy.ndarray): Array of time points.
        u (numpy.ndarray): Coupling term history.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if backend == "numpy":
        np.save(out_path / "x.npy", x)
        if save_y:
            np.save(out_path / "y.npy", y)
        np.save(out_path / "t.npy", t)
    elif backend == "zarr":
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

        save_zarr_array(out_path / "x.zarr", x, compressor)
        if save_y:
            save_zarr_array(out_path / "y.zarr", y, compressor)
        save_zarr_array(out_path / "t.zarr", t, compressor)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")


def save_output_l96_ensemble_split(
    out_path, x, y, t, split, save_y=True, backend="numpy"
):
    """
    Save ensemble simulation results to the specified output path.

    Args:
        out_path (str): Path to the output folder.
        x (numpy.ndarray): Array of slow variables (X).
        y (numpy.ndarray): Array of fast variables (Y).
        t (numpy.ndarray): Array of time points.
        u (numpy.ndarray): Coupling term history.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.
        backend (str, optional): Storage backend, either 'numpy' or 'zarr'. Default is 'numpy'.
    """
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    if backend == "numpy":
        np.save(out_path / f"x_{split}.npy", x)
        if save_y:
            np.save(out_path / f"y_{split}.npy", y)
        np.save(out_path / f"t_{split}.npy", t)
    elif backend == "zarr":
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        save_zarr_array(out_path / f"x_{split}.zarr", x, compressor)
        if save_y:
            save_zarr_array(out_path / f"y_{split}.zarr", y, compressor)
        save_zarr_array(out_path / f"t_{split}.zarr", t, compressor)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")


def load_output_l96_ensemble(load_path, load_y=False, backend="numpy", mmap_mode="r"):
    """
    Load ensemble simulation results from the specified load path.

    Args:
        load_path (str): Path to the L96 ensemble data.
        suffix (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of slow variables (X).
            - y (numpy.ndarray): Array of fast variables (Y).
            - t (numpy.ndarray): Array of time points.
    """
    load_path = Path(load_path)
    if backend == "numpy":
        x = np.load(load_path / "x.npy", mmap_mode=mmap_mode)
        t = np.load(load_path / "t.npy", mmap_mode=mmap_mode)
        if load_y:
            y = np.load(load_path / "y.npy", mmap_mode=mmap_mode)
    elif backend == "zarr":
        x = zarr.open_array(load_path / "x.zarr", mode="r")
        t = zarr.open_array(load_path / "t.zarr", mode="r")
        if load_y:
            y = zarr.open_array(load_path / "y.zarr", mode="r")
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")

    if load_y:
        return x, y, t
    return x, t


def merge_output_l96_ensemble(
    load_path, merge_y=False, backend="numpy", delete_parts=False
):
    if backend == "numpy":
        x_parts = _sorted_parts(load_path, "x_*.npy")
        y_parts = _sorted_parts(load_path, "y_*.npy") if merge_y else []
        t_parts = _sorted_parts(load_path, "t_*.npy")
        x_out = load_path / "x.npy"
        y_out = load_path / "y.npy"
        t_out = load_path / "t.npy"
    elif backend == "zarr":
        x_parts = _sorted_parts(load_path, "x_*.zarr", index_regex=r"_(\d+)\.zarr$")
        y_parts = (
            _sorted_parts(load_path, "y_*.zarr", index_regex=r"_(\d+)\.zarr$")
            if merge_y
            else []
        )
        t_parts = _sorted_parts(load_path, "t_*.zarr", index_regex=r"_(\d+)\.zarr$")
        x_out = load_path / "x.zarr"
        y_out = load_path / "y.zarr"
        t_out = load_path / "t.zarr"
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'numpy' or 'zarr'.")

    if not x_parts or (merge_y and not y_parts) or not t_parts:
        raise FileNotFoundError(f"No data found at {load_path}.")

    if backend == "numpy":
        _stitch_to_numpy(x_parts, x_out)
        _merge_time_parts(t_parts, t_out, backend)
        _stitch_to_numpy(y_parts, y_out)
    elif backend == "zarr":
        asyncio.run(_stitch_to_zarr_async(x_parts, x_out))
        _merge_time_parts(t_parts, t_out, backend)
        asyncio.run(_stitch_to_zarr_async(y_parts, y_out))

    if not delete_parts:
        return

    # Clean up part files
    if x_out.exists():
        _delete_parts(x_parts, backend)
    if merge_y and y_out.exists():
        _delete_parts(y_parts, backend)
    if t_out.exists():
        _delete_parts(t_parts, backend)


def _sorted_parts(load_path, pattern, index_regex=r"_(\d+)\.npy$"):
    files = glob.glob(str(load_path / pattern))
    if not files:
        return []

    def idx(f):
        m = re.search(index_regex, f)
        return int(m.group(1)) if m else float("inf")

    return sorted(files, key=idx)


def _stitch_to_numpy(parts, out_path, axis=0):
    if not parts:
        return
    # Inspect shape/dtype from the first part
    first = np.load(parts[0], mmap_mode="r")
    dtype = first.dtype

    # ---- special case: new axis at 0 ----
    if axis == "new":
        shape = (len(parts),) + tuple(first.shape)
        mm = open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)

        for i, p in enumerate(parts):
            arr = np.load(p, mmap_mode="r")
            mm[i, ...] = arr  # writes into the new axis
        mm.flush()
        return

    # ---- concat along existing axis ----
    shape = list(first.shape)
    # Compute total length along concat axis
    shape[axis] = sum(np.load(p, mmap_mode="r").shape[axis] for p in parts)
    shape = tuple(shape)

    mm = open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
    i = 0
    for p in parts:
        arr = np.load(p, mmap_mode="r")
        n = arr.shape[axis]
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(i, i + n)
        mm[tuple(slc)] = arr  # chunked copy; still bounded memory
        i += n
    mm.flush()


def _merge_time_parts(parts, out_path, backend):
    if not parts:
        return

    first = np.asarray(_open_part_array(parts[0]))
    if first.ndim > 1:
        if backend == "numpy":
            _stitch_to_numpy(parts, out_path)
        else:
            asyncio.run(_stitch_to_zarr_async(parts, out_path))
        return

    for p in parts[1:]:
        arr = np.asarray(_open_part_array(p))
        if arr.shape != first.shape or not np.allclose(arr, first):
            raise ValueError("Time arrays differ across splits; cannot merge.")

    if backend == "numpy":
        np.save(out_path, first)
        return

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
    save_zarr_array(out_path, first, compressor)


async def _stitch_to_zarr_async(parts, out_path, axis=0):
    """Stitch multiple chunks into a single compressed Zarr array.
    axis: int (concat along existing axis) or "new" (create new axis at 0 and stack)
    """
    if not parts:
        return

    first = _open_part_array(parts[0])
    dtype = first.dtype

    # ---- special case: new axis at 0 ----
    if axis == "new":
        shape = (len(parts),) + tuple(first.shape)

        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        store = LocalStore(out_path)

        z = await zarr.api.asynchronous.open_array(
            store=store,
            mode="w",
            shape=shape,
            dtype=dtype,
            compressor=compressor,
            zarr_format=2,
        )

        for i, p in enumerate(parts):
            arr = np.asarray(_open_part_array(p))
            await z.setitem(
                (slice(i, i + 1),) + (slice(None),) * arr.ndim, arr[None, ...]
            )

        store.close()
        return

    # ---- concat along existing axis ----
    if not parts:
        return

    first = _open_part_array(parts[0])
    dtype = first.dtype
    shape = list(first.shape)
    shape[axis] = sum(_open_part_array(p).shape[axis] for p in parts)
    shape = tuple(shape)

    compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)

    # Make the on-disk destination explicit
    store = LocalStore(out_path)

    # Asynchronously create/open the Zarr array
    z = await zarr.api.asynchronous.open_array(
        store=store,
        mode="w",
        shape=shape,
        dtype=dtype,
        compressor=compressor,
        zarr_format=2,
    )

    i = 0
    for p in parts:
        arr = _open_part_array(p)
        n = arr.shape[axis]
        slc = [slice(None)] * arr.ndim
        slc[axis] = slice(i, i + n)
        data = np.asarray(arr)
        await z.setitem(tuple(slc), data)
        i += n

    # Ensure resources are finalized properly
    store.close()


def _open_part_array(path):
    if str(path).endswith(".zarr"):
        return zarr.open_array(path, mode="r")
    return np.load(path, mmap_mode="r")


def _delete_parts(parts, backend):
    for p in parts:
        try:
            if backend == "zarr":
                shutil.rmtree(p)
            else:
                os.remove(p)
        except FileNotFoundError:
            pass
