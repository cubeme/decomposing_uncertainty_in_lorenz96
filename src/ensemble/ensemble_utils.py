"""Validate ensemble shapes and simulation outputs."""

import numpy as np


def validate_init_states_shape(
    config, init_states_: np.ndarray
) -> tuple[int, int, int, int]:
    if init_states_.ndim != 4:
        raise ValueError(f"init_states must be 4D, got shape {init_states_.shape}")

    N, M, L, K = init_states_.shape

    expected = (config.n_init_states, config.n_ens_members, config.n_models, config.K)

    if (N, M, L, K) != expected:
        raise ValueError(
            f"init_states shape {init_states_.shape} does not match expected "
            f"(n_init_states, n_ens_members, n_models, K)={expected}."
        )

    return N, M, L, K


def check_time_consistency(t0, t):
    """
    Check that two time arrays are consistent.

    Args:
        t0 (np.ndarray): Reference time array.
        t (np.ndarray): Time array to compare.

    Raises:
        ValueError: If time arrays differ in shape, start, or end time.
    """
    if t.shape != t0.shape or t[0] != t0[0] or t[-1] != t0[-1]:
        raise ValueError(
            "Time arrays differ across runs; cannot compress to a single t array."
        )


def check_ar_order_consistency(ar_order, rho_arr, sigma):
    if ar_order == 0:
        # AR(0)/iid: rho and sigma are unused but should be consistent
        if rho_arr.shape not in [(), (1,)]:
            raise ValueError(
                f"Config ar_order=0 expects scalar rho (or length-1 placeholder), got rho shape {rho_arr.shape}."
            )
    elif ar_order == 1:
        # AR(1): rho must be scalar (or length-1); sigma optional
        if rho_arr.shape not in [(), (1,)]:
            raise ValueError(
                f"Config ar_order=1 expects scalar rho (or length-1 array), got rho shape {rho_arr.shape}."
            )
        if sigma is None:
            raise ValueError(
                "Config ar_order=1 requires sigma (innovation std) to be provided."
            )

    else:
        # AR(p>1): rho must be length p and sigma must be provided
        if rho_arr.shape != (ar_order,):
            raise ValueError(
                f"Config ar_order={ar_order} expects rho shape ({ar_order},), got rho shape {rho_arr.shape}."
            )
        if sigma is None:
            raise ValueError(
                f"Config ar_order={ar_order} requires sigma (innovation std) to be provided."
            )
