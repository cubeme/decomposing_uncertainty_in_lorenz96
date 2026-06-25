"""Generate Wilks-style local-covariance perturbations."""

import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

import numpy as np


def _wilks_ens_for_one_state(
    x0: np.ndarray,  # (K,)
    x_long: np.ndarray,  # (T, K)
    n_ens: int,
    sigma_clim: float,
    cube_side: float | None = None,
    min_analogs: int = 100,
    expand_factor: float = 1.15,
    max_expands: int = 200,
    rng: np.random.Generator | None = None,
    jitter: float = 1e-10,
) -> np.ndarray:
    """
    Wilks-style ensemble initialization around x0 using local analogue covariance,
    then rescaling so mean marginal std is 0.05*sigma_clim.

    Returns: (n_ens, K)
    """
    rng = np.random.default_rng() if rng is None else rng
    x0 = np.asarray(x0, dtype=float).reshape(-1)
    X = np.asarray(x_long, dtype=float)

    if X.ndim != 2 or X.shape[1] != x0.size:
        raise ValueError(f"Expected x_long shape (T,K) with K={x0.size}, got {X.shape}")

    K = x0.size

    if cube_side is None:
        clim_range = float(np.max(X) - np.min(X))
        cube_side = 0.05 * clim_range
    half = 0.5 * cube_side

    mask = np.max(np.abs(X - x0[None, :]), axis=1) <= half
    n = int(mask.sum())

    expands = 0
    while n < min_analogs and expands < max_expands:
        cube_side *= expand_factor
        half = 0.5 * cube_side
        mask = np.max(np.abs(X - x0[None, :]), axis=1) <= half
        n = int(mask.sum())
        expands += 1

    if n < min_analogs:
        raise RuntimeError(
            f"Only found {n} analogues for one state after expanding to cube_side={cube_side:.4g}. "
            "Increase x_long length, lower min_analogs, or increase max_expands/expand_factor."
        )

    A = X[mask]
    S_local = np.cov(A, rowvar=False, bias=False)

    tr = float(np.trace(S_local))
    if tr <= 0 or not np.isfinite(tr):
        raise RuntimeError(
            "Degenerate S_local (non-positive trace). Try more analogues or larger cube."
        )

    mean_var_local = tr / K
    target_var_mean = (0.05 * float(sigma_clim)) ** 2
    scale = target_var_mean / mean_var_local
    S_init = scale * S_local

    L = np.linalg.cholesky(S_init + jitter * np.eye(K))
    Z = rng.standard_normal(size=(n_ens, K))
    pert = Z @ L.T
    return x0[None, :] + pert


# Globals set in worker initializer (shared memory to avoid copying x_long to each worker)
_X_LONG = None
_X_SHM = None
_X_SHAPE = None
_X_DTYPE = None


def _init_worker_xlong(shm_name: str, shape: tuple[int, int], dtype_str: str):
    """Attach to shared memory in each worker."""
    global _X_LONG, _X_SHM, _X_SHAPE, _X_DTYPE
    _X_SHM = SharedMemory(name=shm_name)
    _X_SHAPE = shape
    _X_DTYPE = np.dtype(dtype_str)
    _X_LONG = np.ndarray(_X_SHAPE, dtype=_X_DTYPE, buffer=_X_SHM.buf)


def _wilks_worker(args):
    """
    args = (i, x0, n_ens, sigma_clim, seed, wilks_kwargs)
    Returns (i, ens_i)
    """
    i, x0, n_ens, sigma_clim, seed, wilks_kwargs = args
    rng = np.random.default_rng(seed)
    ens_i = _wilks_ens_for_one_state(
        x0=x0,
        x_long=_X_LONG,
        n_ens=n_ens,
        sigma_clim=sigma_clim,
        rng=rng,
        **wilks_kwargs,
    )
    return i, ens_i


def perturb_wilks(
    arr: np.ndarray,
    n_ens: int,
    x_long: np.ndarray,
    sigma_clim: float,
    rng: np.random.Generator,
    num_workers: int | None = None,
    chunksize: int = 1,
    **wilks_kwargs,
) -> np.ndarray:
    """
    Parallel Wilks perturbations across initial states.

    - Uses multiprocessing with shared memory so x_long is not copied to each worker.
    - Ensures deterministic results for a given `rng` by generating per-state seeds.

    Parameters:
      arr: (n_states, K)
      n_ens: number of ensemble members
      x_long: (T, K) long run
      sigma_clim: scalar std used in scaling
      rng: np.random.Generator used to generate per-state seeds
      num_workers: default uses min(os.cpu_count(), n_states)
      chunksize: passed to Pool.imap_unordered
      wilks_kwargs: forwarded to _wilks_ens_for_one_state (cube_side, min_analogs, ...)

    Returns:
      out: (n_states, n_ens, K)
    """
    arr = np.asarray(arr, dtype=float)
    X = np.asarray(x_long, dtype=float)

    n_states, K = arr.shape
    if X.ndim != 2 or X.shape[1] != K:
        raise ValueError(f"x_long must have shape (T,{K}). Got {X.shape}.")

    if num_workers is None:
        num_workers = min(mp.cpu_count() or 1, n_states)

    # Put x_long into shared memory
    shm = SharedMemory(create=True, size=X.nbytes)
    try:
        X_sh = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
        X_sh[:] = X  # copy once

        # Deterministic per-state seeds derived from rng
        seeds = rng.integers(0, 2**63 - 1, size=n_states, dtype=np.int64)

        tasks = [
            (i, arr[i].copy(), n_ens, float(sigma_clim), int(seeds[i]), wilks_kwargs)
            for i in range(n_states)
        ]

        out = np.empty((n_states, n_ens, K), dtype=float)

        ctx = mp.get_context("spawn")  # safer across platforms
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_worker_xlong,
            initargs=(shm.name, X.shape, X.dtype.str),
        ) as pool:
            for i, ens_i in pool.imap_unordered(
                _wilks_worker, tasks, chunksize=chunksize
            ):
                out[i] = ens_i

        return out

    finally:
        shm.close()
        shm.unlink()
