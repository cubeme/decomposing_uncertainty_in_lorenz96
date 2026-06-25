import shutil
from pathlib import Path

from run import run_ensemble_l96


def test_run_ensemble_l96_single_output(configs_dir):
    cfg = configs_dir / "l96_ensemble.yaml"
    run_ensemble_l96.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/l96_ensemble_test/")
    assert (out_dir / "ens_l96_init2_mem1" / "x.npy").exists()

    shutil.rmtree(out_dir)


def test_run_ensemble_l96_split_output(configs_dir):
    cfg = configs_dir / "l96_ensemble_split.yaml"
    run_ensemble_l96.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/l96_ensemble_test/")
    assert (out_dir / "ens_l96_init2_mem1" / "x.npy").exists()

    shutil.rmtree(out_dir)
