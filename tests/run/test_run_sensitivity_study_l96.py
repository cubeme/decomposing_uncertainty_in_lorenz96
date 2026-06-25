import shutil
from pathlib import Path

from run import run_sensitivity_study_l96


def test_run_sensitivity_study_l96_output(configs_dir):
    cfg = configs_dir / "l96_sensitivity_study.yaml"
    run_sensitivity_study_l96.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/l96_sensitivity_test/")
    assert (out_dir / "l96" / "x.npy").exists()
    assert (out_dir / "l96" / "y.npy").exists()
    assert (out_dir / "l96" / "t.npy").exists()

    shutil.rmtree(out_dir)
