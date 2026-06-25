import shutil
from pathlib import Path

from run import run_generate_initial_states


def test_run_generate_initial_states(configs_dir):
    cfg = configs_dir / "generate_initial_states.yaml"

    run_generate_initial_states.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/init_states_test/initial_states")
    assert (out_dir / "x.npy").exists()
    assert (out_dir / "y.npy").exists()
    assert (out_dir / "t.npy").exists()

    shutil.rmtree(out_dir.parent)
