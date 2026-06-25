import shutil
from pathlib import Path

from run import run_single_l96


def test_run_single_l96(configs_dir):
    cfg = configs_dir / "single_l96.yaml"

    run_single_l96.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/l96_single_test")
    assert (out_dir / "l96" / "x.zarr").exists()
    assert (out_dir / "l96" / "y.zarr").exists()
    assert (out_dir / "l96" / "t.zarr").exists()

    shutil.rmtree(out_dir)
