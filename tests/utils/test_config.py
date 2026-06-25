from __future__ import annotations

import pytest
import yaml

from models.forcing_schedule import ConstantForcingSchedule
from utils.config import (
    AR_P_PARAMS_DIR_NAME,
    COEFS_DIR_NAME,
    FLOW_MODEL_DIR_NAME,
    L96_SINGLE_OUTPUT_SUBDIR,
    ConfigFlowTraining,
    ConfigGCM,
    ConfigL96,
    ConfigParamsFit,
    ConfigPerturbInitialStates,
)

# -------------------------
# Helpers
# -------------------------


def dump_yaml(temp_dir, name: str, cfg: dict):
    path = temp_dir / name
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def assert_has_attrs(obj, attrs):
    missing = [a for a in attrs if not hasattr(obj, a)]
    assert not missing, f"Missing attributes: {missing}"


# -------------------------
# BaseConfig / shared behavior
# -------------------------


def test_base_fields_load(temp_dir):
    cfg = {
        "experiment_name": "exp",
        "run_module": "run",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "dt_full": 1.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "results_dir": str(temp_dir / "results"),
        "seed": 123,
    }
    path = dump_yaml(temp_dir, "base.yaml", cfg)
    c = ConfigL96(path)

    assert c.experiment_name == "exp"
    assert c.run_module == "run"
    assert c.sweep_name == ""
    assert c.results_dir.name == "results"
    assert c.seed == 123
    assert c.y_scale == 1.0


def test_y_scale_override_loads(temp_dir):
    cfg = {
        "experiment_name": "exp",
        "run_module": "run",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "y_scale": 0.5,
        "f_schedule": {"type": "constant", "F": 20.0},
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "y_scale.yaml", cfg)
    c = ConfigL96(path)

    assert c.y_scale == 0.5


def test_missing_forcing_schedule_raises(temp_dir):
    cfg = {
        "experiment_name": "exp",
        "run_module": "run",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "missing_schedule.yaml", cfg)
    with pytest.raises(ValueError, match="Missing required f_schedule"):
        ConfigL96(path)


def test_f_schedule_and_F_mutually_exclusive(temp_dir):
    cfg = {
        "experiment_name": "exp",
        "run_module": "run",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "F": 20.0,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "both_schedule_and_F.yaml", cfg)
    with pytest.raises(ValueError, match="Specify either f_schedule or F"):
        ConfigL96(path)


@pytest.mark.parametrize("bad_backend", ["", "pickle"])
def test_invalid_backend_raises(temp_dir, bad_backend):
    cfg = {
        "experiment_name": "exp",
        "run_module": "run",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "save_backend": bad_backend,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_backend.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid save_backend"):
        ConfigL96(path)


# -------------------------
# ConfigGCM
# -------------------------


@pytest.fixture
def gcm_det_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_gcm_det",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "baseline_det",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "total_time": 5,
        "dt": 0.01,
        "si": 0.05,
        "time_stepping": "RK2",
        "n_init_states": 10,
        "n_ens_members": 5,
        "init_states_type": "perturbed",
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
        "cpu_count": 2,
        "seed": 42,
    }
    return dump_yaml(temp_dir, "gcm_det.yaml", cfg)


@pytest.fixture
def gcm_flow_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_gcm_flow",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "flow",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "total_time": 5,
        "dt": 0.01,
        "si": 0.05,
        "time_stepping": "RK2",
        "n_init_states": 10,
        "n_models": 3,
        "init_states_type": "perfect",
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_flow_params",
        "results_dir": str(temp_dir / "results"),
        "cpu_count": 2,
        "seed": 42,
        "flow_device": "cpu",
        "noise_type": "white",
        "ar_order": 0,  # required by current logic
    }
    return dump_yaml(temp_dir, "gcm_flow.yaml", cfg)


def test_config_gcm_general_attributes(gcm_det_config_yaml):
    c = ConfigGCM(gcm_det_config_yaml)
    assert c.parameterization_type == "baseline_det"
    assert c.simulation_type == "ensemble"
    assert isinstance(c.f_schedule, ConstantForcingSchedule)

    assert_has_attrs(
        c,
        [
            "experiment_name",
            "run_module",
            "results_dir",
            "seed",
            "parameterization_type",
            "simulation_type",
            "K",
            "J",
            "h",
            "b",
            "c",
            "f_schedule",
            "dt",
            "si",
            "dt_full",
            "total_time",
            "time_stepping",
            "init_states_dir",
            "params_dir",
            "coefs_dir_name",
            "ar_parameters_dir_name",
            "flow_model_dir_name",
        ],
    )


def test_config_gcm_ensemble_attributes(gcm_det_config_yaml):
    c = ConfigGCM(gcm_det_config_yaml)
    assert_has_attrs(
        c,
        [
            "n_init_states",
            "n_ens_members",
            "init_states_type",
            "cpu_count",
            "save_backend",
            "load_backend",
        ],
    )
    assert c.save_backend == "numpy"
    assert c.load_backend == "numpy"


def test_config_gcm_invalid_parameterization_type(temp_dir):
    cfg = {
        "experiment_name": "bad",
        "run_module": "run",
        "simulation_type": "ensemble",
        "parameterization_type": "nope",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "F": 20.0,
        "n_init_states": 1,
        "n_ens_members": 1,
        "init_states_type": "perturbed",
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_param_type.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid parameterization type"):
        ConfigGCM(path)


def test_config_gcm_output_dir_ensemble(gcm_det_config_yaml, temp_dir):
    c = ConfigGCM(gcm_det_config_yaml)
    out = c.output_dir(temp_dir)
    assert str(out).endswith(
        f"ens_gcm_{c.parameterization_type}_init{c.n_init_states}_mem{c.n_ens_members}_models{c.n_models}"
    )


def test_config_gcm_perfect_states_requires_single_member(temp_dir):
    cfg = {
        "experiment_name": "bad_models_members",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "bayesian_regression",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "n_init_states": 3,
        "n_models": 2,
        "n_ens_members": 2,  # invalid when perfect states
        "init_states_type": "perfect",
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_models_members.yaml", cfg)
    with pytest.raises(
        ValueError,
        match="n_ens_members must be 1 when init_states_type='perfect'. Got n_ens_members=2.",
    ):
        ConfigGCM(path)


def test_config_gcm_model_seed_is_loaded_for_model_uncertainty(temp_dir):
    cfg = {
        "experiment_name": "model_seed_range",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "baseline_ar_p",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "F": 20.0,
        "n_init_states": 2,
        "n_models": 3,
        "n_ens_members": 1,
        "init_states_type": "perfect",
        "model_start_seed": 10,
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "model_seed_range.yaml", cfg)
    c = ConfigGCM(path)
    assert c.model_start_seed == 10


def test_config_gcm_flow_special_attributes_and_defaults(
    gcm_flow_config_yaml, temp_dir
):
    c = ConfigGCM(gcm_flow_config_yaml)

    assert c.parameterization_type == "flow"
    assert c.n_models == 3

    assert_has_attrs(c, ["flow_device", "noise_type", "ar_order"])
    assert c.flow_device == "cpu"
    assert c.noise_type == "white"
    assert c.ar_order == 0

    # flow variation defaults
    assert c.use_flexible_tails is False
    assert c.ttf_init_lambda == 0.1
    assert c.delta_t == 0
    assert c.include_forcing_in_cond is False

    out = c.output_dir(temp_dir)
    assert str(out).endswith(
        f"ens_gcm_flow_init{c.n_init_states}_mem{c.n_ens_members}_models{c.n_models}"
    )


def test_config_gcm_flow_invalid_noise_type_raises(temp_dir):
    cfg = {
        "experiment_name": "bad_noise",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "flow",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "n_init_states": 1,
        "n_models": 2,
        "init_states_type": "perfect",
        "n_ens_members": 1,
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
        "flow_device": "cpu",
        "noise_type": "pink",
        "ar_order": 0,
    }
    path = dump_yaml(temp_dir, "bad_noise.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid noise_type"):
        ConfigGCM(path)


def test_config_gcm_flow_white_requires_ar_order_zero(temp_dir):
    cfg = {
        "experiment_name": "white_requires_ar0",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "flow",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "F": 20.0,
        "n_init_states": 1,
        "n_models": 2,
        "init_states_type": "perfect",
        "n_ens_members": 1,
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
        "flow_device": "cpu",
        "noise_type": "white",
        "ar_order": 1,  # invalid per current code
    }
    path = dump_yaml(temp_dir, "white_bad_ar.yaml", cfg)
    with pytest.raises(
        ValueError, match="ar_order must be 0 when noise_type is 'white'"
    ):
        ConfigGCM(path)


def test_config_gcm_flow_ar_p_requires_positive_order(temp_dir):
    cfg = {
        "experiment_name": "ar_p_requires_positive",
        "run_module": "run_ensemble_gcm",
        "simulation_type": "ensemble",
        "parameterization_type": "flow",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "n_init_states": 1,
        "n_models": 2,
        "init_states_type": "perfect",
        "n_ens_members": 1,
        "init_states_dir": "results/test_init_states",
        "params_dir": "results/test_params",
        "results_dir": str(temp_dir / "results"),
        "flow_device": "cpu",
        "noise_type": "ar_p",
        "ar_order": 0,  # invalid
    }
    path = dump_yaml(temp_dir, "arp_bad_order.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid ar_order"):
        ConfigGCM(path)


# -------------------------
# ConfigL96
# -------------------------


@pytest.fixture
def l96_single_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_l96_single",
        "run_module": "run_single_l96",
        "simulation_type": "single",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "F": 20.0,
        "total_time": 10.0,
        "spin_up_time": 5.0,
        "dt": 0.001,
        "si": 0.005,
        "save_backend": "numpy",
        "results_dir": str(temp_dir / "results"),
        "seed": 42,
        "plot": False,
    }
    return dump_yaml(temp_dir, "l96_single.yaml", cfg)


@pytest.fixture
def l96_ensemble_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_l96_ensemble",
        "run_module": "run_ensemble_l96",
        "simulation_type": "ensemble",
        "init_states_type": "perturbed",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "total_time": 5.0,
        "dt": 0.001,
        "si": 0.005,
        "n_init_states": 10,
        "n_ens_members": 2,
        "init_states_dir": "results/test_init_states",
        "results_dir": str(temp_dir / "results"),
        "cpu_count": 2,
        "states_mem_limit": 50,
        "seed": 42,
    }
    return dump_yaml(temp_dir, "l96_ensemble.yaml", cfg)


@pytest.fixture
def l96_sensitivity_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_l96_sensitivity",
        "run_module": "run_sensitivity_study_l96",
        "simulation_type": "sensitivity_study",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "total_time": 5.0,
        "dt": 0.001,
        "si": 0.005,
        "n_init_states": 10,
        "init_states_dir": "results/test_init_states",
        "results_dir": str(temp_dir / "results"),
        "cpu_count": 2,
        "states_mem_limit": 50,
        "seed": 42,
    }
    return dump_yaml(temp_dir, "l96_sensitivity.yaml", cfg)


def test_config_l96_single_loads(l96_single_config_yaml):
    c = ConfigL96(l96_single_config_yaml)
    assert c.simulation_type == "single"
    assert c.spin_up_time == 5.0
    assert c.save_backend == "numpy"
    assert c.load_backend == "numpy"

    assert_has_attrs(
        c,
        [
            "experiment_name",
            "run_module",
            "results_dir",
            "seed",
            "simulation_type",
            "K",
            "J",
            "h",
            "b",
            "c",
            "f_schedule",
            "total_time",
            "dt",
            "si",
            "plot",
            "plot_start_time",
        ],
    )


def test_config_l96_ensemble_loads(l96_ensemble_config_yaml):
    c = ConfigL96(l96_ensemble_config_yaml)
    assert c.simulation_type == "ensemble"
    assert c.init_states_type == "perturbed"
    assert c.n_init_states == 10
    assert c.n_ens_members == 2
    assert c.save_y is True  # default

    assert_has_attrs(
        c,
        [
            "init_states_dir",
            "cpu_count",
            "states_mem_limit",
            "save_backend",
            "load_backend",
            "save_y",
        ],
    )


def test_config_l96_sensitivity_loads(l96_sensitivity_config_yaml):
    c = ConfigL96(l96_sensitivity_config_yaml)
    assert c.simulation_type == "sensitivity_study"
    assert c.n_ens_members == 1  # forced

    assert_has_attrs(
        c,
        [
            "init_states_dir",
            "cpu_count",
            "states_mem_limit",
            "save_backend",
            "load_backend",
        ],
    )


def test_config_l96_ic_generation_selection_defaults(temp_dir):
    cfg = {
        "experiment_name": "test_ic_gen",
        "run_module": "run_ic_generation_l96",
        "simulation_type": "IC_generation",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "n_init_states": 12,
        "generate_method": "selection",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "ic_gen.yaml", cfg)
    c = ConfigL96(path)

    assert c.simulation_type == "IC_generation"
    assert c.generate_method == "selection"
    assert c.selection_mtu == 20  # default
    assert c.spin_up_time == 20.0  # default for IC_generation


def test_config_l96_invalid_simulation_type_raises(temp_dir):
    cfg = {
        "experiment_name": "bad",
        "run_module": "run",
        "simulation_type": "nope",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_sim_type.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid simulation type"):
        ConfigL96(path)


def test_config_l96_perfect_init_requires_single_member(temp_dir):
    cfg = {
        "experiment_name": "perfect_requires_mem1",
        "run_module": "run_ensemble_l96",
        "simulation_type": "ensemble",
        "init_states_type": "perfect",
        "n_init_states": 2,
        "n_ens_members": 3,  # invalid
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "init_states_dir": "results/test_init_states",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "l96_perfect_bad.yaml", cfg)
    with pytest.raises(
        ValueError, match="n_ens_members must be 1 when init_states_type='perfect'"
    ):
        ConfigL96(path)


def test_config_l96_output_dir_variants(
    l96_single_config_yaml, l96_ensemble_config_yaml, temp_dir
):
    single = ConfigL96(l96_single_config_yaml)
    assert str(single.output_dir(temp_dir)).endswith(L96_SINGLE_OUTPUT_SUBDIR)

    ens = ConfigL96(l96_ensemble_config_yaml)
    assert str(ens.output_dir(temp_dir)).endswith(
        f"ens_l96_init{ens.n_init_states}_mem{ens.n_ens_members}"
    )


# -------------------------
# ConfigParamsFit
# -------------------------


@pytest.fixture
def params_fit_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_params_fit",
        "run_module": "run_parameter_fitting_baseline",
        "params_to_fit": ["poly_coefs", "ar_p"],
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "F": 20.0,
        "si": 0.005,
        "poly_order": 3,
        "fit_method": "yule_walker",
        "train_perc": 0.8,
        "results_dir": str(temp_dir / "results"),
        "seed": 42,
    }
    return dump_yaml(temp_dir, "params_fit.yaml", cfg)


def test_config_params_fit_loads(params_fit_config_yaml):
    c = ConfigParamsFit(params_fit_config_yaml)

    assert c.l96_load_backend == "numpy"
    assert c.l96_output_sub_dir == L96_SINGLE_OUTPUT_SUBDIR
    assert "poly_coefs" in c.params_to_fit
    assert "ar_p" in c.params_to_fit
    assert c.poly_order == 3
    assert c.fit_method == "yule_walker"

    assert_has_attrs(
        c,
        [
            "experiment_name",
            "run_module",
            "results_dir",
            "seed",
            "params_to_fit",
            "l96_data_dir",
            "l96_load_backend",
            "l96_output_sub_dir",
            "K",
            "J",
            "h",
            "b",
            "c",
            "f_schedule",
            "si",
            "train_perc",
            "chunk_length",
        ],
    )


def test_config_params_fit_params_to_fit_scalar_becomes_list(temp_dir):
    cfg = {
        "experiment_name": "test_params_fit_scalar",
        "run_module": "run_parameter_fitting_baseline",
        "params_to_fit": "ar_p",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "params_fit_scalar.yaml", cfg)
    c = ConfigParamsFit(path)
    assert c.params_to_fit == ["ar_p"]


def test_config_params_fit_output_dirs(params_fit_config_yaml, temp_dir):
    c = ConfigParamsFit(params_fit_config_yaml)
    assert str(c.coefs_dir(temp_dir)).endswith(COEFS_DIR_NAME)
    assert str(c.ar_parameters_dir(temp_dir)).endswith(AR_P_PARAMS_DIR_NAME)


def test_config_params_fit_invalid_fit_method_raises(temp_dir):
    cfg = {
        "experiment_name": "test_params_fit_invalid_fit_method",
        "run_module": "run_parameter_fitting_baseline",
        "params_to_fit": ["ar_p"],
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "fit_method": "invalid",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "params_fit_invalid_fit_method.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid fit_method"):
        ConfigParamsFit(path)


def test_config_params_fit_bayes_fields_present_only_when_requested(temp_dir):
    cfg = {
        "experiment_name": "test_params_fit_bayes",
        "run_module": "run_parameter_fitting_baseline",
        "params_to_fit": ["bayes_coefs"],
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "params_fit_bayes.yaml", cfg)
    c = ConfigParamsFit(path)

    # defaults should exist
    assert_has_attrs(
        c,
        [
            "poly_order",
            "chains",
            "draws",
            "tune",
            "n_ens_members",
            "n_models",
            "chunk_length",
        ],
    )


# -------------------------
# ConfigPerturbInitialStates
# -------------------------


@pytest.fixture
def perturb_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_perturb",
        "run_module": "run_perturb_initial_states",
        "init_states_dir": "results/test_init_states",
        "perturb_iid": True,
        "perturb_std": 0.1,
        "n_init_states": 10,
        "n_ens_members": 2,
        "results_dir": str(temp_dir / "results"),
        "seed": 42,
    }
    return dump_yaml(temp_dir, "perturb.yaml", cfg)


def test_config_perturb_loads(perturb_config_yaml):
    c = ConfigPerturbInitialStates(perturb_config_yaml)
    assert c.perturb_std == 0.1
    assert c.n_init_states == 10
    assert c.n_ens_members == 2
    assert str(c.init_states_dir) == "results/test_init_states"

    assert_has_attrs(
        c,
        [
            "experiment_name",
            "run_module",
            "results_dir",
            "seed",
            "init_states_dir",
            "conditional_params",
            "perturb_std",
            "n_init_states",
            "n_ens_members",
        ],
    )
    assert isinstance(c.conditional_params, dict)


def test_config_perturb_defaults_n_init_states_none(temp_dir):
    cfg = {
        "experiment_name": "test_perturb_defaults",
        "run_module": "run_perturb_initial_states",
        "init_states_dir": "results/test_init_states",
        "perturb_iid": True,
        "perturb_std": 0.1,
        "n_ens_members": 2,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "perturb_defaults.yaml", cfg)
    c = ConfigPerturbInitialStates(path)
    assert c.n_init_states is None


def test_config_perturb_requires_iid_or_wilks(temp_dir):
    cfg = {
        "experiment_name": "test_perturb_missing_flags",
        "run_module": "run_perturb_initial_states",
        "init_states_dir": "results/test_init_states",
        "perturb_std": 0.1,
        "n_ens_members": 2,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "perturb_missing_flags.yaml", cfg)
    with pytest.raises(
        ValueError, match="Must specify either perturb_iid or perturb_wilks"
    ):
        ConfigPerturbInitialStates(path)


def test_config_perturb_rejects_iid_and_wilks_together(temp_dir):
    cfg = {
        "experiment_name": "test_perturb_both_flags",
        "run_module": "run_perturb_initial_states",
        "init_states_dir": "results/test_init_states",
        "perturb_iid": True,
        "perturb_wilks": True,
        "perturb_std": 0.1,
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "n_ens_members": 2,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "perturb_both_flags.yaml", cfg)
    with pytest.raises(ValueError, match="Cannot use both IID and Wilks"):
        ConfigPerturbInitialStates(path)


def test_config_perturb_wilks_loads(temp_dir):
    cfg = {
        "experiment_name": "test_perturb_wilks",
        "run_module": "run_perturb_initial_states",
        "init_states_dir": "results/test_init_states",
        "perturb_wilks": True,
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "cpu_count": 4,
        "n_init_states": 5,
        "n_ens_members": 3,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "perturb_wilks.yaml", cfg)
    c = ConfigPerturbInitialStates(path)
    assert str(c.l96_data_dir) == "results/test_l96_data"
    assert c.l96_load_backend == "numpy"
    assert c.cpu_count == 4


# -------------------------
# ConfigFlowTraining
# -------------------------


@pytest.fixture
def flow_training_config_yaml(temp_dir):
    cfg = {
        "experiment_name": "test_flow_training",
        "run_module": "run_flow_training",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "n_coupling_layers": 4,
        "hidden_dims": [32, 32],
        "train_perc": 0.7,
        "val_perc": 0.2,
        "test_perc": 0.1,
        "chunk_length": 10,
        "devices": 1,
        "strategy": None,
        "num_workers_data_loader": 0,
        "results_dir": str(temp_dir / "results"),
        "fit_ar_parameters": True,
        "ar_order": 1,
        "fit_method": "yule_walker",
        "seed": 42,
    }
    return dump_yaml(temp_dir, "flow_training.yaml", cfg)


def test_config_flow_training_loads(flow_training_config_yaml, temp_dir):
    c = ConfigFlowTraining(flow_training_config_yaml)
    assert c.l96_load_backend == "numpy"
    assert c.l96_output_sub_dir == L96_SINGLE_OUTPUT_SUBDIR
    assert c.n_coupling_layers == 4
    assert c.hidden_dims == [32, 32]
    assert c.fit_ar_parameters is True
    assert c.ar_order == 1
    assert c.fit_method == "yule_walker"
    assert c.base_dist == "gaussian"

    assert_has_attrs(
        c,
        [
            "experiment_name",
            "run_module",
            "results_dir",
            "seed",
            "l96_data_dir",
            "K",
            "J",
            "h",
            "b",
            "c",
            "f_schedule",
            "si",
            "l96_load_backend",
            "l96_output_sub_dir",
            "n_coupling_layers",
            "hidden_dims",
            "epochs",
            "batch_size",
            "lr",
            "weight_decay",
            "grad_clip",
            "train_perc",
            "val_perc",
            "test_perc",
            "chunk_length",
            "device",
            "devices",
            "strategy",
            "num_workers_data_loader",
            "tensorboard_log_dir",
            "early_stopping_patience",
            "early_stopping_min_delta",
            "early_stopping_monitor",
            "base_dist",
            "seq_len",
            "fit_ar_parameters",
            "ar_order",
            "fit_method",
        ],
    )

    # variations defaults
    assert c.use_flexible_tails is False
    assert c.ttf_init_lambda == 0.1
    assert c.delta_t == 0
    assert c.include_forcing_in_cond is False

    # dir helpers
    assert str(c.output_dir(temp_dir)).endswith(FLOW_MODEL_DIR_NAME)
    assert str(c.ar_parameters_dir(temp_dir)).endswith(AR_P_PARAMS_DIR_NAME)
    assert str(c.coefs_dir(temp_dir)).endswith(COEFS_DIR_NAME)
    assert c.chunk_length == 10


def test_config_flow_training_invalid_split_sum_raises(temp_dir):
    cfg = {
        "experiment_name": "bad_split",
        "run_module": "run_flow_training",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "n_coupling_layers": 4,
        "hidden_dims": [32, 32],
        "train_perc": 0.8,
        "val_perc": 0.2,
        "test_perc": 0.1,  # sum > 1
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_split.yaml", cfg)
    with pytest.raises(ValueError, match=r"train_perc \+ val_perc \+ test_perc"):
        ConfigFlowTraining(path)


def test_config_flow_training_invalid_base_dist_raises(temp_dir):
    cfg = {
        "experiment_name": "bad_base_dist",
        "run_module": "run_flow_training",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "n_coupling_layers": 4,
        "hidden_dims": [32, 32],
        "base_dist": "laplace",
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "bad_base_dist.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid base_dist"):
        ConfigFlowTraining(path)


def test_config_flow_training_ar_p_base_loads(temp_dir):
    cfg = {
        "experiment_name": "flow_training_arp",
        "run_module": "run_flow_training",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "n_coupling_layers": 4,
        "hidden_dims": [32, 32],
        "base_dist": "ar_p",
        "ar_order": 2,
        "init_rho": [0.1, -0.02],
        "init_sigma": 0.8,
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "flow_training_arp.yaml", cfg)
    c = ConfigFlowTraining(path)
    assert c.base_dist == "ar_p"
    assert c.ar_order == 2
    assert c.init_rho == [0.1, -0.02]
    assert c.init_sigma == 0.8


def test_config_flow_training_ar_p_base_invalid_init_rho_length_raises(temp_dir):
    cfg = {
        "experiment_name": "flow_training_arp_bad_rho",
        "run_module": "run_flow_training",
        "l96_data_dir": "results/test_l96_data",
        "l96_load_backend": "numpy",
        "K": 8,
        "J": 32,
        "h": 1.0,
        "b": 10.0,
        "c": 10.0,
        "f_schedule": {"type": "constant", "F": 20.0},
        "si": 0.005,
        "n_coupling_layers": 4,
        "hidden_dims": [32, 32],
        "base_dist": "ar_p",
        "ar_order": 3,
        "init_rho": [0.1, -0.02],
        "results_dir": str(temp_dir / "results"),
    }
    path = dump_yaml(temp_dir, "flow_training_arp_bad_rho.yaml", cfg)
    with pytest.raises(ValueError, match="Invalid init_rho length"):
        ConfigFlowTraining(path)
