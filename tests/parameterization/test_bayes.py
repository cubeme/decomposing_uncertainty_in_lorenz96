from parameterization.bayes.compute_bayesian_regression import (
    fit_bayesian_regression,
)


def test_fit_bayesian_regression_smoke(sample_x, sample_u, poly_order):
    # tiny settings to keep it fast
    return_samples = 3
    coefs = fit_bayesian_regression(
        sample_x[:10],
        sample_u[:10],
        poly_order,
        chains=1,
        draws=5,
        tune=5,
        return_samples=return_samples,
    )
    assert coefs.shape == (return_samples, poly_order + 1)
