import numpy as np

from perturbations.iid import perturb_iid


def test_perturb_iid_shape_and_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    n_states, K, n_ens = 7, 8, 20
    std = 0.02
    arr = np.zeros((n_states, K))

    out1 = perturb_iid(arr, n_ens=n_ens, std=std, rng=rng1)
    out2 = perturb_iid(arr, n_ens=n_ens, std=std, rng=rng2)

    assert out1.shape == (n_states, n_ens, K)
    assert np.allclose(out1, out2)


def test_perturb_iid_mean_near_input_for_large_ens():
    rng = np.random.default_rng(0)

    n_states, K, n_ens = 3, 8, 2000
    std = 0.1
    arr = np.linspace(-1.0, 1.0, n_states * K).reshape(n_states, K)

    out = perturb_iid(arr, n_ens=n_ens, std=std, rng=rng)
    ens_mean = out.mean(axis=1)

    # With large ensemble, sample mean should be close to arr.
    # Tolerance is loose to avoid flakiness.
    assert np.allclose(ens_mean, arr, atol=1e-2, rtol=0.0)
