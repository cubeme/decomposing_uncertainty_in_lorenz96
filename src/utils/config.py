"""Load and validate experiment configurations."""

import multiprocessing as mp
from pathlib import Path

import yaml
from absl import logging

from models.forcing_schedule import parse_forcing_schedule

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent

L96_SINGLE_OUTPUT_SUBDIR = "l96"
COEFS_DIR_NAME = "coefs"
FLOW_MODEL_DIR_NAME = "flow_model"
AR_P_PARAMS_DIR_NAME = "ar_parameters"


class BaseConfig:
    def __init__(self, yaml_config):
        with open(yaml_config, "r") as f:
            self._config_data = yaml.safe_load(f)

        self.experiment_name = self._config_data["experiment_name"]
        self.run_module = self._config_data["run_module"]
        self.sweep_name = self._config_data.get("sweep_name", "")

        # Set results directory
        self.results_dir = Path(self._config_data.get("results_dir", ROOT / "results"))

        # Miscellaneous parameters
        self.seed = self._config_data.get("seed", 0)  # random seed

        # Full time step in L96
        self.dt_full = self._config_data.get("dt_full", 1.0)

    def _validate_backend(self, backend, name):
        if backend not in {"numpy", "zarr"}:
            raise ValueError(f"Invalid {name}: {backend}. Must be 'numpy' or 'zarr'.")

    def _set_l96_params(self, config_data, eval_mode=False):
        # Lorenz '96 model parameters
        self.K = config_data["K"]  # number of slow variables
        self.J = config_data["J"]  # number of fast variables per slow variable
        self.h = config_data["h"]  # coupling coefficient
        self.b = config_data["b"]  # spatial-scale ratio
        self.c = config_data["c"]  # time-scale ratio
        # scaling factor for fast variables in coupling term for when we are converting between L96 formulations
        self.y_scale = config_data.get("y_scale", 1.0)

        if (
            config_data.get("f_schedule") is not None
            and config_data.get("F") is not None
        ):
            raise ValueError("Specify either f_schedule or F in config, not both.")
        # Forcing schedule
        if "f_schedule" in config_data:
            if isinstance(config_data["f_schedule"], list) and eval_mode:
                self.f_schedule = config_data[
                    "f_schedule"
                ]  # we'll allow lists of schedules in eval mode
            else:
                self.f_schedule = parse_forcing_schedule(config_data["f_schedule"])
                self.f_schedule_data = config_data["f_schedule"]
        elif "F" in config_data:
            if isinstance(config_data["F"], list) and eval_mode:
                self.f_schedule = config_data[
                    "F"
                ]  # we'll allow lists of Fs in eval mode
            else:
                self.f_schedule = parse_forcing_schedule(config_data["F"])
                self.f_schedule_data = {"type": "constant", "F": config_data["F"]}
        else:
            raise ValueError("Missing required f_schedule in config.")

    def _load_ensemble_shape(self, config_data):
        self.init_states_type = config_data["init_states_type"]  # perfect or perturbed
        if self.init_states_type not in ["perfect", "perturbed"]:
            raise ValueError(
                f"Invalid init_states_type: {self.init_states_type}. Must be 'perfect' or 'perturbed'."
            )
        # Number of initial states
        self.n_init_states = config_data["n_init_states"]  # number of initial states

        # Number of ensemble members per perturbed initial states
        if self.init_states_type == "perturbed" and "n_ens_members" not in config_data:
            raise ValueError(
                "n_ens_members is required when init_states_type='perturbed'."
            )
        self.n_ens_members = config_data.get("n_ens_members", 1)

        if self.init_states_type == "perfect" and self.n_ens_members != 1:
            raise ValueError(
                f"n_ens_members must be 1 when init_states_type='perfect'. Got n_ens_members={self.n_ens_members}."
            )

        self.n_models = config_data.get(
            "n_models", 1
        )  # number of model realizations for model uncertainty ensembles


class ConfigGCM(BaseConfig):
    def __init__(self, yaml_config, eval_mode=False):
        super().__init__(yaml_config)
        config_data = self._config_data
        # We can relax some of the checks in eval mode
        self.eval_mode = eval_mode

        parameterization_types = [
            "none",
            "baseline_det",
            "baseline_ar_p",
            "bayesian_regression",
            "flow",
        ]
        if config_data["parameterization_type"] not in parameterization_types:
            raise ValueError(
                f"Invalid parameterization type: {config_data['parameterization_type']}. "
                f"Must be one of {parameterization_types}."
            )
        self.parameterization_type = config_data["parameterization_type"]
        self.simulation_type = config_data["simulation_type"]

        # Settings of the corresponding L96 model
        self._set_l96_params(config_data, eval_mode=eval_mode)
        self.load_sweep = config_data.get("load_sweep", {})

        # Simulation time parameters
        self.total_time = config_data.get("total_time", 100.0)

        # Time stepping parameters
        self.dt = config_data.get("dt", 0.001)  # time step
        self.si = config_data.get("si", 0.005)  # sampling interval
        self.time_stepping = config_data.get(
            "time_stepping", "RK2"
        )  # time stepping function

        # Load directories
        self.init_states_dir = (
            Path(config_data["init_states_dir"])
            if not self.eval_mode
            else config_data.get("init_states_dir", None)
        )
        self.params_dir = Path(
            config_data["params_dir"]
        )  # directory with saved parameterization parameters

        # Ensemble simulation parameters
        if self.simulation_type == "ensemble":
            self._load_ensemble_shape(config_data)
            if self.parameterization_type == "baseline_det":
                if not self.n_models == 1:
                    logging.info(
                        "Baseline deterministic parameterization does not support multiple model realizations. Setting n_models to 1."
                    )
                    self.n_models = 1  # no model uncertainty ensembles for deterministic parameterizations

            self.cpu_count = config_data.get(
                "cpu_count", mp.cpu_count()
            )  # number of CPU cores to use
            # Save/load backend
            self.save_backend = config_data.get("save_backend", "numpy")
            self.load_backend = config_data.get("load_backend", self.save_backend)
            self._validate_backend(self.save_backend, "save_backend")
            self._validate_backend(self.load_backend, "load_backend")

            # Stochastic models can run ensembles per model realization to estimate model uncertainty
            if self.parameterization_type in [
                "baseline_ar_p",
                "bayesian_regression",
                "flow",
            ]:
                # Model seed settings
                self.model_start_seed = config_data.get("model_start_seed", 1)

        if self.parameterization_type == "flow":
            self.flow_device = config_data.get("flow_device", "cpu")
            self.noise_type = config_data.get("noise_type", "white")
            # Check noise_type
            allowed = {"white", "ar_p"}
            nt = self.noise_type
            # normalize to list
            if isinstance(nt, (list, tuple)):
                values = nt
            else:
                values = [nt]

            # validate
            if not all(v in allowed for v in values):
                raise ValueError(
                    f"Invalid noise_type: {self.noise_type}. "
                    f"Allowed values are {sorted(allowed)}."
                )

            # AR(p) parameters
            self.ar_order = config_data.get("ar_order", 1)
            # Validate only if not in eval mode where we may load sweep for ar_order and noise_type)
            if not self.eval_mode:
                if self.noise_type == "ar_p":
                    if not isinstance(self.ar_order, int) or self.ar_order < 1:
                        raise ValueError(
                            f"Invalid ar_order: {self.ar_order}. Must be a positive integer."
                        )
                if self.noise_type == "white" and self.ar_order != 0:
                    raise ValueError(
                        f"ar_order must be 0 when noise_type is 'white'. Got ar_order={self.ar_order}."
                    )

            # Variations for normal flow
            self.use_flexible_tails = config_data.get(
                "use_flexible_tails", False
            )  # whether to use Tail Transform Flow
            self.ttf_init_lambda = config_data.get(
                "ttf_init_lambda", 0.1
            )  # initial lambda for TTF
            self.delta_t = config_data.get(
                "delta_t", 0
            )  # number of past steps in condition
            self.include_forcing_in_cond = config_data.get(
                "include_forcing_in_cond", False
            )  # whether to include forcing in condition

        # Baseline AR(p) parameterization parameters
        if self.parameterization_type == "baseline_ar_p":
            self.ar_order = config_data.get("ar_order", 1)  # order of AR(p) process

        # Directories to load parameterization parameters
        self.coefs_dir_name = COEFS_DIR_NAME
        self.ar_parameters_dir_name = AR_P_PARAMS_DIR_NAME
        self.flow_model_dir_name = FLOW_MODEL_DIR_NAME

    def output_dir(self, base_dir):
        return (
            Path(base_dir)
            / f"ens_gcm_{self.parameterization_type}_init{self.n_init_states}_mem{self.n_ens_members}_models{self.n_models}"
        )
        return Path(base_dir) / f"gcm_{self.parameterization_type}"


class ConfigL96(BaseConfig):
    def __init__(self, yaml_config, eval_mode=False):
        super().__init__(yaml_config)
        config_data = self._config_data
        # We can relax some of the checks in eval mode
        self.eval_mode = eval_mode

        simulation_types = [
            "single",
            "ensemble",
            "spin_up",
            "IC_generation",
            "sensitivity_study",
        ]
        if config_data["simulation_type"] not in simulation_types:
            raise ValueError(
                f"Invalid simulation type: {config_data['simulation_type']}. "
                f"Must be one of {simulation_types}."
            )
        self.simulation_type = config_data["simulation_type"]

        # Lorenz '96 model parameters
        self._set_l96_params(config_data, eval_mode=eval_mode)

        self.total_time = config_data.get("total_time", 100.0)

        # Simulation type parameters
        if self.simulation_type in ["single", "spin_up", "IC_generation"]:
            self.spin_up_time = config_data.get("spin_up_time", 20.0)

        if self.simulation_type == "single":
            self.save_backend = config_data.get("save_backend", "numpy")
            self.load_backend = config_data.get("load_backend", self.save_backend)
            self._validate_backend(self.save_backend, "save_backend")
            self._validate_backend(self.load_backend, "load_backend")

        if self.simulation_type == "spin_up":
            self.number_of_spin_ups = config_data["number_of_spin_ups"]

        if self.simulation_type == "IC_generation":
            self.n_init_states = config_data["n_init_states"]
            self.generate_method = config_data[
                "generate_method"
            ]  # method to generate initial conditions ("spin_up" or "selection")
            if self.generate_method == "selection":
                self.selection_mtu = config_data.get(
                    "selection_mtu", 20
                )  # step size for selection method in MTU

        # Parameters for ensemble and sensitivity study simulations
        if self.simulation_type in ["ensemble", "sensitivity_study"]:
            self.init_states_dir = (
                Path(config_data["init_states_dir"])
                if not self.eval_mode
                else config_data.get("init_states_dir", None)
            )
            self.n_init_states = config_data[
                "n_init_states"
            ]  # number of initial states

            self.cpu_count = config_data.get(
                "cpu_count", mp.cpu_count()
            )  # number of CPU cores to use
            self.states_mem_limit = config_data.get(
                "states_mem_limit", 100
            )  # number of ensemble states that can fit into memory
            # Save/load backend
            self.save_backend = config_data.get("save_backend", "numpy")
            self.load_backend = config_data.get("load_backend", self.save_backend)
            self._validate_backend(self.save_backend, "save_backend")
            self._validate_backend(self.load_backend, "load_backend")

        if self.simulation_type == "ensemble":
            self._load_ensemble_shape(config_data)
            self.save_y = config_data.get("save_y", True)

        if self.simulation_type == "sensitivity_study":
            self.n_ens_members = 1

        # Time stepping parameters
        self.dt = config_data.get("dt", 0.001)  # time step
        self.si = config_data.get("si", 0.005)  # sampling interval

        # Plotting parameters
        self.plot = config_data.get("plot", True)  # whether to plot results
        self.plot_start_time = config_data.get(
            "plot_start_time", 0.0
        )  # time to start plotting

    def output_dir(self, base_dir):
        if self.simulation_type == "ensemble":
            return (
                Path(base_dir)
                / f"ens_l96_init{self.n_init_states}_mem{self.n_ens_members}"
            )
        if self.simulation_type in ["single", "sensitivity_study"]:
            return Path(base_dir) / L96_SINGLE_OUTPUT_SUBDIR
        if self.simulation_type == "spin_up":
            return Path(base_dir) / "l96_spin_up"

        return Path(base_dir)


class ConfigParamsFit(BaseConfig):
    def __init__(self, yaml_config, eval_mode=False):
        super().__init__(yaml_config)
        config_data = self._config_data
        self.eval_mode = eval_mode

        # List of parameters to fit
        self.params_to_fit = config_data[
            "params_to_fit"
        ]  # poly_coefs, ar_p, bayes_coefs
        if not isinstance(self.params_to_fit, list):
            self.params_to_fit = [self.params_to_fit]
        allowed_params = {"poly_coefs", "ar_p", "bayes_coefs"}
        for param in self.params_to_fit:
            if param not in allowed_params:
                raise ValueError(
                    f"Invalid parameter to fit: {param}. Must be one of {allowed_params}."
                )

        # Lorenz '96 model parameters
        self.l96_data_dir = Path(config_data["l96_data_dir"])  # directory with L96 data
        self._set_l96_params(config_data, eval_mode=self.eval_mode)
        self.si = config_data["si"]  # sampling interval used in data
        self.l96_load_backend = config_data[
            "l96_load_backend"
        ]  # backend used to save the data
        self._validate_backend(self.l96_load_backend, "l96_load_backend")
        # This setting depends ConfigL96.output_dir for simulation_type="single"
        self.l96_output_sub_dir = L96_SINGLE_OUTPUT_SUBDIR

        # Fitting parameters
        if "poly_coefs" in self.params_to_fit or "bayes_coefs" in self.params_to_fit:
            self.poly_order = config_data.get("poly_order", 3)  # polynomial order
        if "ar_p" in self.params_to_fit:
            self.ar_order = config_data.get("ar_order", 1)  # order of AR(p) process
            self.fit_method = config_data.get("fit_method", "least_squares")
            if self.fit_method not in ["yule_walker", "least_squares"]:
                raise ValueError(
                    f"Invalid fit_method: {self.fit_method}. Must be 'yule_walker' or 'least_squares'."
                )

        if "bayes_coefs" in self.params_to_fit:
            self.chains = config_data.get("chains", 4)  # number of chains
            self.draws = config_data.get("draws", 1000)  # number of draws
            self.tune = config_data.get("tune", 2000)  # number of tuning steps
            self.n_ens_members = config_data.get(
                "n_ens_members", 20
            )  # number of ensemble members represented in sampled coefficients
            self.n_models = config_data.get(
                "n_models", 1
            )  # number of model realizations represented in sampled coefficients

        self.train_perc = config_data.get(
            "train_perc", 1.0
        )  # percentage of data used for training
        self.chunk_length = config_data.get(
            "chunk_length", 100.0
        )  # chunk length in time units for chunk sampling when training time variant forcing schedules

        self.plot = config_data.get("plot", True)  # whether to plot results

    def coefs_dir(self, path):
        return Path(path) / COEFS_DIR_NAME

    def ar_parameters_dir(self, path):
        return Path(path) / AR_P_PARAMS_DIR_NAME


class ConfigPerturbInitialStates(BaseConfig):
    def __init__(self, yaml_config):
        super().__init__(yaml_config)
        config_data = self._config_data

        self.init_states_dir = Path(
            config_data["init_states_dir"]
        )  # directory with initial states
        self.load_sweep = config_data.get("load_sweep", {})
        self.conditional_params = config_data.get("conditional_params", {})

        self.perturb_iid = config_data.get(
            "perturb_iid", False
        )  # whether to use IID Gaussian perturbations
        self.perturb_wilks = config_data.get(
            "perturb_wilks", False
        )  # whether to use Wilks perturbations (local analogue covariance)

        if self.perturb_iid and self.perturb_wilks:
            raise ValueError(
                "Cannot use both IID and Wilks perturbations. Please choose one."
            )
        if not (self.perturb_iid ^ self.perturb_wilks):
            raise ValueError(
                "Must specify either perturb_iid or perturb_wilks as True."
            )

        if self.perturb_iid:
            self.perturb_std = config_data[
                "perturb_std"
            ]  # standard deviation of perturbation

        if self.perturb_wilks:
            # Paths to long runs used to find analogues
            self.l96_data_dir = Path(
                config_data["l96_data_dir"]
            )  # directory with L96 data
            self.l96_load_backend = config_data[
                "l96_load_backend"
            ]  # backend used to save the data
            self._validate_backend(self.l96_load_backend, "l96_load_backend")
            self.l96_output_sub_dir = L96_SINGLE_OUTPUT_SUBDIR
            self.cpu_count = config_data.get(
                "cpu_count", mp.cpu_count()
            )  # number of CPU cores to use

        self.n_init_states = config_data.get(
            "n_init_states", None
        )  # number of states to perturb (None = all)
        self.n_ens_members = config_data[
            "n_ens_members"
        ]  # number of perturbations per initial state


class ConfigFlowTraining(BaseConfig):
    def __init__(self, yaml_config, eval_mode=False):
        super().__init__(yaml_config)
        config_data = self._config_data
        self.eval_mode = eval_mode

        # Variations for normal flow
        self.use_flexible_tails = config_data.get(
            "use_flexible_tails", False
        )  # whether to use Tail Transform Flow
        self.ttf_init_lambda = config_data.get(
            "ttf_init_lambda", 0.1
        )  # initial lambda for TTF
        self.delta_t = config_data.get(
            "delta_t", 0
        )  # number of past steps in condition
        self.include_forcing_in_cond = config_data.get(
            "include_forcing_in_cond", False
        )  # whether to include forcing in condition
        self.base_dist = config_data.get(
            "base_dist", "gaussian"
        )  # base distribution for flow
        if self.base_dist not in ["gaussian", "ar_p"]:
            raise ValueError(
                f"Invalid base_dist: {self.base_dist}. Must be 'gaussian' or 'ar_p'."
            )
        if self.base_dist == "ar_p":
            self.ar_order = config_data.get("ar_order", 1)  # order of AR(p) process
            if not isinstance(self.ar_order, int) or self.ar_order < 1:
                raise ValueError(
                    f"Invalid ar_order: {self.ar_order}. Must be a positive integer."
                )
            self.init_rho = config_data.get(
                "init_rho", [0.0] * self.ar_order
            )  # initial value for AR coefficients (if ar_p base is used)
            if isinstance(self.init_rho, (int, float)):
                if self.ar_order != 1:
                    raise ValueError(
                        f"ar_order must be 1 when init_rho is a single number. Got ar_order={self.ar_order} and init_rho={self.init_rho}."
                    )
            elif len(self.init_rho) != self.ar_order:
                raise ValueError(
                    f"Invalid init_rho length: {len(self.init_rho)}. Must match ar_order={self.ar_order}."
                )
            self.init_sigma = config_data.get(
                "init_sigma", 1.0
            )  # initial value for AR noise std (if ar_p base is used)
            if self.init_sigma <= 0:
                raise ValueError(f"Invalid init_sigma: {self.init_sigma}. Must be > 0.")
        else:
            # AR(p) fit order for post-training diagnostics (when fit_ar_parameters=True)
            self.ar_order = config_data.get("ar_order", 1)
            if isinstance(self.ar_order, int):
                if self.ar_order < 1:
                    raise ValueError(
                        f"Invalid ar_order: {self.ar_order}. Must be a positive integer."
                    )
            elif isinstance(self.ar_order, list):
                if len(self.ar_order) == 0 or not all(
                    isinstance(v, int) and v > 0 for v in self.ar_order
                ):
                    raise ValueError(
                        f"Invalid ar_order list: {self.ar_order}. Must be a non-empty list of positive integers."
                    )
            else:
                raise ValueError(
                    f"Invalid ar_order: {self.ar_order}. Must be a positive integer or list of positive integers."
                )

        self.seq_len = config_data.get("seq_len", None)  # sequence length for training
        if self.seq_len is None:
            if self.base_dist == "ar_p":
                self.seq_len = max(self.ar_order + 1, 8)
            else:
                self.seq_len = 1
        if self.seq_len <= 0:
            raise ValueError(
                f"Invalid seq_len: {self.seq_len}. Must be a positive integer."
            )

        # L96 data source
        self.l96_data_dir = Path(config_data["l96_data_dir"])
        self.load_sweep = config_data.get("load_sweep", {})

        # Lorenz '96 model parameters
        self._set_l96_params(config_data, eval_mode=self.eval_mode)
        self.si = config_data["si"]  # sampling interval used in data
        self.l96_load_backend = config_data[
            "l96_load_backend"
        ]  # backend used to save the data
        self._validate_backend(self.l96_load_backend, "l96_load_backend")
        # This setting depends ConfigL96.output_dir for simulation_type="single"
        self.l96_output_sub_dir = L96_SINGLE_OUTPUT_SUBDIR

        # Flow model parameters
        self.n_coupling_layers = config_data["n_coupling_layers"]
        self.hidden_dims = config_data["hidden_dims"]

        # Training parameters
        self.epochs = config_data.get("epochs", 50)
        self.batch_size = config_data.get("batch_size", 256)
        self.lr = config_data.get("lr", 1e-3)

        self.weight_decay = config_data.get("weight_decay", 0.0)
        self.grad_clip = config_data.get("grad_clip", None)

        self.train_perc = config_data.get("train_perc", 1.0)
        self.val_perc = config_data.get("val_perc", 0.0)
        self.test_perc = config_data.get("test_perc", 0.0)
        if self.train_perc + self.val_perc + self.test_perc > 1.0:
            raise ValueError("train_perc + val_perc + test_perc must sum to <= 1.")
        self.chunk_length = config_data.get(
            "chunk_length", 100.0
        )  # chunk length in time units for chunk sampling when training time variant forcing schedules

        self.device = config_data.get("device", "cpu")
        self.devices = config_data.get("devices", 1)
        self.strategy = config_data.get("strategy", None)
        self.num_workers_data_loader = config_data.get("num_workers_data_loader", 0)

        self.tensorboard_log_dir = config_data.get("tensorboard_log_dir", None)
        self.early_stopping_patience = config_data.get("early_stopping_patience", 0)
        self.early_stopping_min_delta = float(
            config_data.get("early_stopping_min_delta", 0.0)
        )
        self.early_stopping_monitor = config_data.get("early_stopping_monitor", "val")

        # Fit AR(p) parameters rho after training if requested
        self.fit_ar_parameters = config_data.get("fit_ar_parameters", False)
        self.fit_method = config_data.get("fit_method", "least_squares")
        if self.fit_method not in ["yule_walker", "least_squares"]:
            raise ValueError(
                f"Invalid fit_method: {self.fit_method}. Must be 'yule_walker' or 'least_squares'."
            )

    def output_dir(self, base_dir):
        return Path(base_dir) / FLOW_MODEL_DIR_NAME

    def ar_parameters_dir(self, base_dir):
        return Path(base_dir) / AR_P_PARAMS_DIR_NAME

    def coefs_dir(self, base_dir):
        return Path(base_dir) / COEFS_DIR_NAME
