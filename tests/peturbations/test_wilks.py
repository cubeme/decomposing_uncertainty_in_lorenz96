import multiprocessing as mp

import numpy as np
import pytest

from perturbations.wilks import perturb_wilks


@pytest.fixture(scope="module", autouse=True)
def _mp_start_method_spawn():
    """
    Make tests consistent across OSes.
    If the start method was already set elsewhere, ignore.
    """
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass


def test_perturb_wilks_shape_and_reproducible():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    n_states, K, n_ens = 5, 8, 30

    # Long run: roughly stationary-ish synthetic sample
    x_long = np.random.default_rng(1).standard_normal(size=(20000, K))

    # Compute sigma_clim the same way you'd do in code
    sigma_clim = float(np.std(x_long, ddof=1))

    arr = np.random.default_rng(2).standard_normal(size=(n_states, K))

    out1 = perturb_wilks(
        arr,
        n_ens=n_ens,
        x_long=x_long,
        sigma_clim=sigma_clim,
        rng=rng1,
        num_workers=2,  # keep tests lightweight + deterministic
        chunksize=1,
    )
    out2 = perturb_wilks(
        arr,
        n_ens=n_ens,
        x_long=x_long,
        sigma_clim=sigma_clim,
        rng=rng2,
        num_workers=2,
        chunksize=1,
    )

    assert out1.shape == (n_states, n_ens, K)
    assert np.allclose(out1, out2)


def test_perturb_wilks_nontrivial_spread():
    rng = np.random.default_rng(7)

    n_states, K, n_ens = 4, 8, 200

    x_long = np.random.default_rng(9).standard_normal(size=(30000, K))
    sigma_clim = float(np.std(x_long, ddof=1))

    arr = np.zeros((n_states, K))
    out = perturb_wilks(
        arr,
        n_ens=n_ens,
        x_long=x_long,
        sigma_clim=sigma_clim,
        rng=rng,
        num_workers=2,
        chunksize=1,
    )

    assert np.std(out) > 0.0


def test_perturb_wilks_target_std_is_reasonable_order():
    """
    Wilks rescales so that the *mean marginal std per component* is ~ 0.05*sigma_clim,
    but with finite analogues and finite ensemble size we only check the order of magnitude.
    """
    rng = np.random.default_rng(11)

    n_states, K, n_ens = 3, 8, 800
    x_long = np.random.default_rng(5).standard_normal(size=(50000, K))
    sigma_clim = float(np.std(x_long, ddof=1))
    target = 0.05 * sigma_clim

    arr = np.random.default_rng(6).standard_normal(size=(n_states, K))
    out = perturb_wilks(
        arr,
        n_ens=n_ens,
        x_long=x_long,
        sigma_clim=sigma_clim,
        rng=rng,
        num_workers=2,
        chunksize=1,
    )

    pert = out - arr[:, None, :]
    std_est = np.std(pert, axis=1, ddof=1)  # (n_states, K)
    mean_std_est = float(std_est.mean())

    assert 0.5 * target <= mean_std_est <= 2.0 * target


def test_perturb_wilks_raises_on_bad_x_long_shape():
    rng = np.random.default_rng(0)
    arr = np.zeros((2, 8))
    x_long_bad = np.zeros((1000, 7))  # wrong K
    with pytest.raises(ValueError):
        perturb_wilks(
            arr,
            n_ens=10,
            x_long=x_long_bad,
            sigma_clim=1.0,
            rng=rng,
            num_workers=2,
        )
