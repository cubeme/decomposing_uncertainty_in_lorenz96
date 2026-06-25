"""Generate IID Gaussian initial-condition perturbations."""

import numpy as np


def perturb_iid(
    arr: np.ndarray, n_ens: int, std: float, rng: np.random.Generator
) -> np.ndarray:
    """
    arr: (n_states, K) -> (n_states, n_ens, K)
    """
    arr = np.asarray(arr)
    noise = rng.normal(0.0, float(std), size=(arr.shape[0], n_ens, arr.shape[1]))
    return arr[:, None, :] + noise
