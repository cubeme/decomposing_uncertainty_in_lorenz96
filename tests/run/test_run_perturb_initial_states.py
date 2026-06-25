import shutil
from pathlib import Path

from run import run_perturb_initial_states


def test_run_perturb_initial_states_iid(configs_dir):
    cfg = configs_dir / "perturb_initial_states_iid.yaml"
    run_perturb_initial_states.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/perturb_init_test/")
    assert (out_dir / "initial_states" / "x.npy").exists()
    assert (out_dir / "initial_states" / "y.npy").exists()
    assert (out_dir / "initial_states" / "t.npy").exists()

    shutil.rmtree(out_dir.parent)


def test_run_perturb_initial_states_wilks(configs_dir):
    cfg = configs_dir / "perturb_initial_states_wilks.yaml"
    run_perturb_initial_states.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/perturb_init_test/")
    assert (out_dir / "initial_states" / "x.npy").exists()
    assert (out_dir / "initial_states" / "y.npy").exists()
    assert (out_dir / "initial_states" / "t.npy").exists()

    shutil.rmtree(out_dir.parent)
