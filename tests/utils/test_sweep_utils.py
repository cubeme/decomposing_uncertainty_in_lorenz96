import pytest
import yaml
from pytest import mark

from utils.sweep_utils import generate_run_configs, keep_only_load_sweep

# ============================================================================
# Tests for generate_run_configs - L96 Ensemble
# ============================================================================


def test_generate_run_configs_l96_ensemble_sweep_f_c(configs_dir, output_root):
    """Test generate_run_configs with L96 ensemble F x c parameter sweep."""
    cfg_path = configs_dir / "l96_ensemble_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Should generate 4 configs (2 F values × 2 c values)
    assert len(configs) == 4
    assert sweep == {
        "c": [4.0, 10.0],
        "F": [18.0, 20.0],
    }

    # Verify all combinations exist
    combinations = {(cfg["c"], cfg["F"]) for cfg in configs}
    expected = {(4.0, 18.0), (4.0, 20.0), (10.0, 18.0), (10.0, 20.0)}
    assert combinations == expected


def test_generate_run_configs_preserves_sweep_names_unique(configs_dir, output_root):
    """Test that generate_run_configs creates unique output names."""
    cfg_path = configs_dir / "l96_ensemble_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # All output names should be unique
    sweep_names = [cfg["sweep_name"] for cfg in configs]
    assert len(set(sweep_names)) == len(sweep_names)


def test_generate_run_configs_sweep_dict_matches_configs(configs_dir, output_root):
    """Test that sweep dictionary from generate_run_configs matches generated configs."""
    cfg_path = configs_dir / "l96_ensemble_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Sweep dict should list all swept parameters
    assert set(sweep.keys()) == {"c", "F"}

    # All configs should have the swept parameters
    for cfg in configs:
        for key in sweep.keys():
            assert key in cfg

    # Verify all values in sweep appear in configs
    all_f_values = {cfg["F"] for cfg in configs}
    all_c_values = {cfg["c"] for cfg in configs}
    assert all_f_values == set(sweep["F"])
    assert all_c_values == set(sweep["c"])


# ============================================================================
# Tests for generate_run_configs - GCM Ensemble
# ============================================================================


def test_generate_run_configs_gcm_baseline_det_sweep_f(configs_dir, output_root):
    """Test generate_run_configs with GCM baseline det F parameter sweep."""
    cfg_path = configs_dir / "gcm_baseline_det_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Should generate 3 configs
    assert len(configs) == 3
    assert sweep == {"F": [19.0, 20.0, 21.0]}

    assert [cfg["F"] for cfg in configs] == [19.0, 20.0, 21.0]


def test_generate_run_configs_gcm_baseline_ar1_ignore_sweep_c(configs_dir, output_root):
    """Test generate_run_configs with GCM AR1 c parameter sweep."""
    cfg_path = configs_dir / "gcm_baseline_ar1_sweep_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert sweep == {"c": [4.0, 10.0]}
    assert len(configs) == 2
    assert [cfg["c"] for cfg in configs] == [4.0, 10.0]
    assert all(cfg["experiment_name"] == "gcm_baseline_ar1_ensemble" for cfg in configs)
    assert all(cfg["f_schedule"]["F"] == 20.0 for cfg in configs)


def test_generate_run_configs_gcm_bayes_sweep_f_c(configs_dir, output_root):
    """Test generate_run_configs with GCM Bayes F x c parameter sweep."""
    cfg_path = configs_dir / "gcm_bayes_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Should generate 4 configs (2 F × 2 c)
    assert len(configs) == 4
    assert sweep == {
        "c": [8.0, 10.0],
        "f_schedule": [
            {"type": "oscillating", "Fmean": 19.0, "amp": 2.0, "freq": 0.5},
            {"type": "oscillating", "Fmean": 20.0, "amp": 2.0, "freq": 0.5},
        ],
    }

    combinations = {(cfg["c"], cfg["f_schedule"]["Fmean"]) for cfg in configs}
    expected = {(8.0, 19.0), (8.0, 20.0), (10.0, 19.0), (10.0, 20.0)}
    assert combinations == expected

    # Verify all F and c values are present
    c_values = {cfg["c"] for cfg in configs}
    assert c_values == {8.0, 10.0}
    f_values = {cfg["f_schedule"]["Fmean"] for cfg in configs}
    assert f_values == {19.0, 20.0}


def test_generate_run_configs_gcm_bayes_n_models(configs_dir, output_root):
    """Test generate_run_configs with GCM Bayes and n_models > 1."""
    cfg_path = configs_dir / "gcm_bayes_n_models_no_sweep.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert sweep == {}
    assert len(configs) == 1
    assert configs[0]["experiment_name"] == "gcm_bayes_ensemble"
    assert configs[0]["sweep_name"] == ""
    assert "model_idx" not in configs[0]


def test_generate_run_configs_gcm_flow_n_models(configs_dir, output_root):
    cfg_path = configs_dir / "gcm_flow_n_models_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert sweep == {
        "f_schedule": [
            {"type": "linear", "F0": 19.0, "F1": 23.0, "t0": 0.0, "t1": 5.0},
            {"type": "linear", "F0": 20.0, "F1": 23.0, "t0": 0.0, "t1": 5.0},
        ]
    }
    assert len(configs) == 2
    assert all(
        cfg["experiment_name"] == "gcm_flow_n_models_ensemble" for cfg in configs
    )


def test_generate_run_configs_preserves_non_swept_fields(configs_dir, output_root):
    """Test that generate_run_configs preserves non-swept config fields."""
    cfg_path = configs_dir / "gcm_bayes_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Non-swept fields should be preserved
    preserved_fields = ["experiment_name", "simulation_type", "parameterization_type"]

    for cfg in configs:
        for field in preserved_fields:
            assert field in cfg
            assert cfg[field] == base_config[field]


# ============================================================================
# Tests for generate_run_configs - Perturb Initial States
# ============================================================================


def test_generate_run_configs_perturb_sweep_std_seeds(configs_dir, output_root):
    """Test generate_run_configs with perturb perturb_std×n_seeds sweep."""
    cfg_path = configs_dir / "perturb_initial_states_sweep_std_seeds.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    # Should generate 4 configs (2 perturb_std × 2 n_seeds)
    assert len(configs) == 4
    assert sweep == {"perturb_std": [0.01, 0.05], "n_seeds": [5, 10]}

    combinations = {(cfg["perturb_std"], cfg["n_seeds"]) for cfg in configs}
    expected = {(0.01, 5), (0.01, 10), (0.05, 5), (0.05, 10)}
    assert combinations == expected


def test_generate_run_configs_conditional_params_applied(configs_dir, output_root):
    """Test that conditional_params updates target values based on sweep key."""
    cfg_path = configs_dir / "perturb_initial_states_conditional_params.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 4
    assert sweep == {"c": [4.0, 10.0], "F": [18.0, 20.0]}
    assert base_config["load_sweep"] == {"c": [4.0, 10.0], "F": [18.0, 20.0]}

    by_c = {cfg["c"]: cfg["perturb_std"] for cfg in configs}
    assert by_c == {4.0: 0.6, 10.0: 0.02}


def test_generate_run_configs_schedule_conditional_params(configs_dir, output_root):
    """Test f_schedule sweep and conditional_params parsing."""
    cfg_path = configs_dir / "l96_ensemble_sweep_schedule_conditional_params.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    expected_schedule = [
        {"type": "linear", "F0": 18, "F1": 23, "t0": 0, "t1": 5},
        {"type": "oscillating", "Fmean": 20, "amp": 2, "freq": 5},
    ]

    assert len(configs) == 2
    assert sweep == {"f_schedule": expected_schedule}
    assert [cfg["f_schedule"] for cfg in configs] == expected_schedule
    for cfg in configs:
        if cfg["f_schedule"]["type"] == "linear":
            assert cfg["init_states_dir"] == "tests/run/data/l96_init_linear"
        elif cfg["f_schedule"]["type"] == "oscillating":
            assert cfg["init_states_dir"] == "tests/run/data/l96_init_oscillating"


# ============================================================================
# Tests for generate_run_configs - Parameter Fitting
# ============================================================================


@mark.parametrize(
    "config_name",
    [
        "parameter_fitting_baseline_sweep_f.yaml",
        "parameter_fitting_bayes_sweep_c.yaml",
    ],
)
def test_generate_run_configs_excludes_params_to_fit_from_sweep(
    config_name, configs_dir, output_root
):
    """Test that generate_run_configs does not sweep params_to_fit list."""
    cfg_path = configs_dir / config_name
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert "params_to_fit" not in sweep
    assert all("params_to_fit" in cfg for cfg in configs)


# ============================================================================
# Tests for generate_run_configs - Flow Training
# ============================================================================


def test_generate_run_configs_flow_hidden_dims_not_swept(configs_dir, output_root):
    """Test that hidden_dims list is not treated as a sweep."""
    cfg_path = configs_dir / "fit_params_flow_no_sweep_hidden_dims.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 1
    assert configs[0]["seed"] == 0
    assert sweep == {}


def test_generate_run_configs_flow_init_rho_single_list_not_swept(output_root):
    base_config = {
        "results_dir": str(output_root),
        "base_dist": "ar_p",
        "ar_order": 2,
        "init_rho": [0.2, -0.1],  # single rho vector, same as hidden_dims=[..]
        "seed": 0,
    }

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 1
    assert sweep == {}
    assert configs[0]["init_rho"] == [0.2, -0.1]


def test_generate_run_configs_flow_init_rho_list_of_lists_swept(output_root):
    base_config = {
        "results_dir": str(output_root),
        "base_dist": "ar_p",
        "ar_order": 2,
        "init_rho": [[0.2, -0.1], [0.1, 0.05]],  # sweep over rho vectors
        "seed": 0,
    }

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 2
    assert sweep == {"init_rho": [[0.2, -0.1], [0.1, 0.05]]}
    assert {tuple(cfg["init_rho"]) for cfg in configs} == {(0.2, -0.1), (0.1, 0.05)}


# ============================================================================
# Tests for generate_run_configs - Noise/ar_order coupling
# ============================================================================


def _noise_pairs(configs):
    return {(cfg["noise_type"], cfg["ar_order"]) for cfg in configs}


def test_generate_run_configs_noise_ar_order_mixed_sweep():
    base_config = {"noise_type": ["white", "ar_p"], "ar_order": [1, 3]}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {
        ("white", 0),
        ("ar_p", 1),
        ("ar_p", 3),
    }


def test_generate_run_configs_noise_list_default_ar_p():
    base_config = {"noise_type": ["white", "ar_p"]}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {
        ("white", 0),
        ("ar_p", 1),
    }


def test_generate_run_configs_white_noise_with_ar_order_errors():
    base_config = {"noise_type": "white", "ar_order": [1, 3]}
    with pytest.raises(ValueError, match="ar_order is only used"):
        generate_run_configs(base_config)


def test_generate_run_configs_ar_p_only_with_ar_order_list():
    base_config = {"noise_type": "ar_p", "ar_order": [1, 3]}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {
        ("ar_p", 1),
        ("ar_p", 3),
    }


def test_generate_run_configs_ar_p_list_defaults_to_one():
    base_config = {"noise_type": ["ar_p"]}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {("ar_p", 1)}


def test_generate_run_configs_white_list_defaults_to_zero():
    base_config = {"noise_type": ["white"]}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {("white", 0)}


def test_generate_run_configs_ar_p_no_sweep():
    base_config = {"noise_type": "ar_p", "ar_order": 3}
    configs, _ = generate_run_configs(base_config)

    assert _noise_pairs(configs) == {("ar_p", 3)}


def test_generate_run_configs_ar_order_list_no_sweep():
    base_config = {"ar_order": [1, 3]}
    configs, _ = generate_run_configs(base_config)

    assert len(configs) == 1
    assert configs[0]["ar_order"] == [1, 3]


# ============================================================================
# Tests for generate_run_configs - Generate Initial States
# ============================================================================


def test_generate_run_configs_generate_initial_states_sweep_n_init(
    configs_dir, output_root
):
    """Test generate_run_configs with initial states n_init_states sweep."""
    cfg_path = configs_dir / "generate_initial_states_sweep_n_init.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 3
    assert sweep == {"n_init_states": [2, 5, 10]}
    assert [cfg["n_init_states"] for cfg in configs] == [2, 5, 10]


# ============================================================================
# Tests for generate_run_configs - Single L96
# ============================================================================


def test_generate_run_configs_single_l96_sweep_f(configs_dir, output_root):
    """Test generate_run_configs with single L96 F parameter sweep."""
    cfg_path = configs_dir / "single_l96_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 2
    assert sweep == {"F": [20.0, 21.0]}
    assert [cfg["F"] for cfg in configs] == [20.0, 21.0]


# ============================================================================
# Tests for generate_run_configs - Forcing Schedule Types
# ============================================================================


def test_generate_run_configs_linear_schedule_sweep(configs_dir, output_root):
    """Test linear forcing schedule sweep expands F1 x t0."""
    cfg_path = configs_dir / "forcing_schedule_linear_sweep.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 4
    assert sweep == {
        "f_schedule": [
            {"type": "linear", "F0": 18, "F1": 20, "t0": 0, "t1": 10},
            {"type": "linear", "F0": 18, "F1": 20, "t0": 5, "t1": 10},
            {"type": "linear", "F0": 18, "F1": 23, "t0": 0, "t1": 10},
            {"type": "linear", "F0": 18, "F1": 23, "t0": 5, "t1": 10},
        ]
    }


def test_generate_run_configs_oscillating_schedule_sweep(configs_dir, output_root):
    """Test oscillating forcing schedule sweep expands amp."""
    cfg_path = configs_dir / "forcing_schedule_oscillating_sweep.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 2
    assert sweep == {
        "f_schedule": [
            {"type": "oscillating", "Fmean": 18, "amp": 2, "freq": 0.5},
            {"type": "oscillating", "Fmean": 18, "amp": 5, "freq": 0.5},
        ]
    }


def test_generate_run_configs_mixed_schedule_list(configs_dir, output_root):
    """Test mixed schedule list expands per-entry sweeps."""
    cfg_path = configs_dir / "forcing_schedule_list_mixed.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    base_config["results_dir"] = str(output_root)

    configs, sweep = generate_run_configs(base_config)

    assert len(configs) == 3
    assert sweep == {
        "f_schedule": [
            {"type": "constant", "F": 19},
            {"type": "linear", "F0": 18, "F1": 20, "t0": 0, "t1": 10},
            {"type": "linear", "F0": 18, "F1": 20, "t0": 5, "t1": 10},
        ]
    }


# ============================================================================
# Tests for keep_only_load_sweep
# ============================================================================


def test_keep_only_load_sweep_keeps_matching_tokens_in_order():
    """Keep matching load_sweep tokens and preserve order."""
    sweep_name = "hidden_dims_8_8-c_10.0-F_20.0-seed_1"
    load_sweep = {"F": [18.0, 20.0], "c": [5.0, 10.0]}

    assert keep_only_load_sweep(sweep_name, load_sweep) == "c_10.0-F_20.0"


def test_keep_only_load_sweep_no_matches_returns_empty():
    """Return empty string when nothing matches."""
    sweep_name = "c_3.0-F_17.0"
    load_sweep = {"F": [18.0], "c": [4.0]}

    assert keep_only_load_sweep(sweep_name, load_sweep) == ""


def test_keep_only_load_sweep_hidden_dims_value_with_hyphen_no_match():
    """Hyphenated hidden_dims values do not match due to token splitting."""
    sweep_name = "hidden_dims_8_8"
    load_sweep = {"hidden_dims": [[8, 8], [16, 16]]}

    assert keep_only_load_sweep(sweep_name, load_sweep) == "hidden_dims_8_8"
