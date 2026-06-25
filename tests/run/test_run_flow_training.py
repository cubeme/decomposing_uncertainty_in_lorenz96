import shutil
from pathlib import Path

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.storage import load_checkpoint
from run import run_flow_training


def test_run_flow_training(configs_dir):
    cfg = configs_dir / "flow_training.yaml"

    run_flow_training.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/flow_training_test")

    assert (out_dir / "flow_model" / "checkpoint.pt").exists()

    ar_orders = [1, 3]
    for ar in ar_orders:
        assert (out_dir / "ar_parameters" / f"rho_{ar}.npy").exists()
        assert (out_dir / "ar_parameters" / f"sigma_{ar}.npy").exists()

    shutil.rmtree(out_dir.parent)


def test_run_flow_training_tail_history_forcing(configs_dir):
    cfg = configs_dir / "flow_training_tail_history_forcing.yaml"

    run_flow_training.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/flow_training_tail_history_forcing_test")
    ar_order = 1
    assert (out_dir / "flow_model" / "checkpoint.pt").exists()
    assert (out_dir / "ar_parameters" / f"rho_{ar_order}.npy").exists()
    assert (out_dir / "ar_parameters" / f"sigma_{ar_order}.npy").exists()

    shutil.rmtree(out_dir.parent)


def test_run_flow_training_ar_p_base(configs_dir):
    cfg = configs_dir / "flow_training_ar_p_base.yaml"

    run_flow_training.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/flow_training_ar_p_base_test")
    flow_dir = out_dir / "flow_model"
    assert (flow_dir / "checkpoint.pt").exists()
    assert (out_dir / "ar_parameters" / "rho_2.npy").exists()
    assert (out_dir / "ar_parameters" / "sigma_2.npy").exists()

    model, checkpoint = load_checkpoint(flow_dir)
    assert isinstance(model.base, ARpBase)
    assert checkpoint["cfg"]["base_dist_name"] == "ar_p"
    assert checkpoint["cfg"]["ar_order"] == 2

    shutil.rmtree(out_dir.parent)
