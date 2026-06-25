import shutil
from pathlib import Path

import numpy as np

from run import run_parameter_fitting_baseline, run_parameter_fitting_bayes
from utils.config import ConfigParamsFit


def test_run_parameter_fitting_baseline(configs_dir):
    cfg = configs_dir / "parameter_fitting_baseline.yaml"

    run_parameter_fitting_baseline.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/param_fit_baseline_test/")
    assert (out_dir / "coefs" / "coefs.npy").exists()
    assert (out_dir / "ar_parameters" / "rho_1.npy").exists()
    assert (out_dir / "ar_parameters" / "sigma_1.npy").exists()

    shutil.rmtree(out_dir.parent)


def test_run_parameter_fitting_baseline_ar2(configs_dir):
    cfg = configs_dir / "parameter_fitting_baseline_ar2.yaml"

    run_parameter_fitting_baseline.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/param_fit_baseline_test/")
    assert (out_dir / "coefs" / "coefs.npy").exists()
    assert (out_dir / "ar_parameters" / "rho_2.npy").exists()
    assert (out_dir / "ar_parameters" / "sigma_2.npy").exists()

    shutil.rmtree(out_dir.parent)


def test_run_parameter_fitting_bayes(configs_dir):
    cfg = configs_dir / "parameter_fitting_bayes.yaml"
    cfg_obj = ConfigParamsFit(cfg)

    run_parameter_fitting_bayes.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/param_fit_bayes_test/")
    coefs_path = out_dir / "coefs" / "bayesian_coefs.npy"
    assert coefs_path.exists()
    coefs = np.load(coefs_path)
    assert coefs.shape == (
        cfg_obj.n_ens_members,
        cfg_obj.n_models,
        cfg_obj.poly_order + 1,
    )

    shutil.rmtree(out_dir.parent)
