import shutil
from pathlib import Path

import numpy as np
from pytest import mark

from run import run_ensemble_gcm
from utils.config import ConfigGCM


def _load_outputs(out_dir, cfg):
    cfg = ConfigGCM(cfg)

    out_dir = Path(cfg.output_dir(out_dir))
    assert (out_dir / "x.npy").exists()
    assert (out_dir / "t.npy").exists()
    x = np.load(out_dir / "x.npy")
    t = np.load(out_dir / "t.npy")
    return cfg, out_dir, x, t


def _assert_outputs(out_dir, cfg, expect_seeds=False):
    cfg_obj, out_dir, _, _ = _load_outputs(out_dir, cfg)
    if expect_seeds:
        seeds_path = out_dir.parent / "seeds" / "seeds.npy"
        assert seeds_path.exists()
        seeds = np.load(seeds_path)
        assert seeds.shape == (
            cfg_obj.n_init_states,
            cfg_obj.n_ens_members,
            cfg_obj.n_models,
        )


def test_baseline_det(configs_dir):
    cfg = configs_dir / "gcm_baseline_det.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_baseline_det_test")
    _assert_outputs(out_dir, cfg)
    shutil.rmtree(out_dir)


def test_baseline_ar_p(configs_dir):
    cfg = configs_dir / "gcm_baseline_ar2.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_baseline_ar2_test")
    _assert_outputs(out_dir, cfg, expect_seeds=True)
    shutil.rmtree(out_dir.parent)


def test_bayesian(configs_dir):
    cfg = configs_dir / "gcm_bayes.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_bayes_test")
    _assert_outputs(out_dir, cfg)
    shutil.rmtree(out_dir)


def test_flow(configs_dir):
    cfg = configs_dir / "gcm_flow.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_flow_test")
    _assert_outputs(out_dir, cfg, expect_seeds=True)
    shutil.rmtree(out_dir)


def test_flow_tail(configs_dir):
    cfg = configs_dir / "gcm_flow_tail.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_flow_tail_test")
    _assert_outputs(out_dir, cfg, expect_seeds=True)
    shutil.rmtree(out_dir)


def test_flow_history_forcing(configs_dir):
    cfg = configs_dir / "gcm_flow_history_forcing.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_flow_history_forcing_test")
    _assert_outputs(out_dir, cfg, expect_seeds=True)
    shutil.rmtree(out_dir)


def test_flow_arp_base(configs_dir):
    cfg = configs_dir / "gcm_flow_arp_base.yaml"
    run_ensemble_gcm.run_from_config_path(str(cfg))

    out_dir = Path("tests/run/output/gcm_flow_arp_base_test")
    _assert_outputs(out_dir, cfg, expect_seeds=True)
    shutil.rmtree(out_dir)


def test_flow_loads_rho_sigma_for_non_white_noise(configs_dir, monkeypatch):
    cfg = configs_dir / "gcm_flow_history_forcing.yaml"
    called = {"load_ar_p_parameters": False}

    def _load_ar_p_parameters_stub(path, ar_order):
        called["load_ar_p_parameters"] = True
        return 0.2, 0.98

    def _run_flow_stub(
        init_states,
        config,
        flow_model,
        seeds,
        rho,
        sigma,
        device,
        time_stepping,
        is_model_uncertainty_ensemble=False,
    ):
        nt = int(config.total_time / config.si) + 1
        n_init, n_members, n_models, k = init_states.shape
        x = np.zeros((n_init, n_members, n_models, nt, k), dtype=np.float32)
        t = np.linspace(0.0, config.total_time, nt)
        return x, t

    monkeypatch.setattr(
        run_ensemble_gcm, "load_ar_p_parameters", _load_ar_p_parameters_stub
    )
    monkeypatch.setattr(run_ensemble_gcm, "run_flow_ensemble", _run_flow_stub)

    run_ensemble_gcm.run_from_config_path(str(cfg))

    assert called["load_ar_p_parameters"] is True
    out_dir = Path("tests/run/output/gcm_flow_history_forcing_test")
    shutil.rmtree(out_dir)


@mark.parametrize(
    "config_name, output_dir_name",
    [
        ("gcm_baseline_ar1_n_models.yaml", "gcm_baseline_ar1_n_models_test"),
        ("gcm_bayes_n_models.yaml", "gcm_bayes_n_models_test"),
        ("gcm_flow_n_models.yaml", "gcm_flow_n_models_test"),
    ],
)
def test_gcm_n_models(configs_dir, config_name, output_dir_name):
    cfg = configs_dir / config_name
    run_ensemble_gcm.run_from_config_path(str(cfg))

    base_out_dir = Path("tests/run/output") / output_dir_name
    cfg_obj, out_dir, x, _ = _load_outputs(base_out_dir, cfg)

    nt = int(cfg_obj.total_time / cfg_obj.si) + 1
    assert x.shape == (
        cfg_obj.n_init_states,
        cfg_obj.n_ens_members,
        cfg_obj.n_models,
        nt,
        cfg_obj.K,
    )

    if "bayes" in config_name:
        coefs = np.load(out_dir.parent / "bayesian_coefs.npy")

        assert coefs.shape[0] == cfg_obj.n_ens_members
        assert coefs.shape[1] == cfg_obj.n_models
        assert np.all(coefs == coefs[0])
        shutil.rmtree(out_dir.parent)
    else:  # ar1 or flow types
        seeds = np.load(out_dir.parent / "seeds" / "seeds.npy")
        expected = np.arange(
            cfg_obj.model_start_seed,
            cfg_obj.model_start_seed
            + (cfg_obj.n_init_states * cfg_obj.n_ens_members * cfg_obj.n_models),
            dtype=int,
        ).reshape(cfg_obj.n_init_states, cfg_obj.n_ens_members, cfg_obj.n_models)
        assert np.all(seeds == expected)
        shutil.rmtree(out_dir.parent)


@mark.parametrize(
    "n_init_states,n_ens_members,n_models,model_start_seed",
    [
        (2, 1, 1, 3),
        (2, 3, 1, 5),
        (3, 1, 2, 7),
        (2, 2, 3, 11),
    ],
)
def test_generate_model_seeds_shape_and_values(
    n_init_states, n_ens_members, n_models, model_start_seed
):
    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.n_init_states = n_init_states
    cfg.n_ens_members = n_ens_members
    cfg.n_models = n_models
    cfg.model_start_seed = model_start_seed

    seeds = run_ensemble_gcm._generate_model_seeds(cfg)

    assert seeds.shape == (n_init_states, n_ens_members, n_models)
    if n_ens_members > 1 and n_models > 1:
        expected_models = np.arange(
            model_start_seed,
            model_start_seed + n_models,
            dtype=int,
        )
        expected = np.broadcast_to(
            expected_models[None, None, :],
            (n_init_states, n_ens_members, n_models),
        )
    else:
        expected = np.arange(
            model_start_seed,
            model_start_seed + (n_init_states * n_ens_members * n_models),
            dtype=int,
        ).reshape(n_init_states, n_ens_members, n_models)
    assert np.array_equal(seeds, expected)


def test_process_bayesian_coefficients_slices_to_config_shape():
    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.n_ens_members = 2
    cfg.n_models = 3

    coefs = np.ones((4, 5, 3))
    out = run_ensemble_gcm._process_bayesian_coefficients(coefs, cfg)
    assert out.shape == (2, 3, 3)


def test_process_initial_states_perturbed_broadcasts_model_axis():
    class DummyConfig:
        pass

    cfg = DummyConfig()
    cfg.n_init_states = 2
    cfg.n_ens_members = 3
    cfg.n_models = 2
    cfg.K = 4
    cfg.init_states_type = "perturbed"
    cfg.init_states_dir = "dummy"

    x_init = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    out = run_ensemble_gcm._process_initial_states(x_init, cfg)
    assert out.shape == (2, 3, 2, 4)
    assert np.array_equal(out[:, :, 0, :], x_init)
    assert np.array_equal(out[:, :, 1, :], x_init)
