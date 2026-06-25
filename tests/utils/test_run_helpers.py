import yaml
from pytest import mark

from utils.run_helpers import determine_run_module

# ============================================================================
# Tests for determine_run_module
# ============================================================================


def test_determine_run_module_l96_ensemble(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "l96_ensemble_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_gcm_baseline_det(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "gcm_baseline_det_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_gcm_baseline_ar1(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "gcm_baseline_ar1_sweep_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_gcm_bayes(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "gcm_bayes_sweep_f_c.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_gcm_flow(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "gcm_flow_n_models_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_perturb_initial_states(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "perturb_initial_states_sweep_std_seeds.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


@mark.parametrize(
    "config_name",
    [
        "parameter_fitting_baseline_sweep_f.yaml",
        "parameter_fitting_bayes_sweep_c.yaml",
    ],
)
def test_determine_run_module_parameter_fitting(config_name, configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / config_name
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_generate_initial_states(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "generate_initial_states_sweep_n_init.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_single_l96(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "single_l96_sweep_f.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"


def test_determine_run_module_flow_training(configs_dir):
    """Test determine_run_module uses explicit config value."""
    cfg_path = configs_dir / "fit_params_flow_no_sweep_hidden_dims.yaml"
    with open(cfg_path) as f:
        base_config = yaml.safe_load(f)

    module = determine_run_module(base_config)
    assert module == f"run.{base_config['run_module']}"
