import numpy as np
import pandas as pd
from evaluation.scripts.script_utils import (
    load_yaml,
    parse_model_meta_from_cfg_key,
    to_numpy,
)
from notebook_utils import generate_sweep_dict_list

from ensemble.storage import load_output_gcm_ensemble, load_output_l96_ensemble
from models.forcing_schedule import forcing_schedule_to_dict
from utils.config import ConfigGCM, ConfigL96
from utils.sweep_utils import get_sweep_name

DEFAULT_SWEEP_KEY = "default"


def compute_full_perfect_metrics(x_arr, x_arr_perfect, x_true):
    """
    x_arr : [N_init, N_ens, N_models, T, K]
    x_arr_perfect : [N_init, T, K]
    x_true : [N_init, T, K]

    Compute metrics for single spatial index k.

    Returns dict of arrays (all shape [T, K]):

    var_perfect: Variance of perfect model runs over (init).
    var_true: Variance of truth over (init).

    spread_total : Var over (init, ens, model).
    spread_all_avg_i : Mean_i Var_{j,m}(X | i).
    spread_ens_avg_i : Mean_i Var_j(E_{m}[X | i]).
    spread_models_avg_i : Mean_i Var_m(E_{j}[X | i]).
    interaction_avg_i : Mean_i interaction term between ensemble and model.
    sum_components_avg_i : Sum of ensemble, model, interaction, and k components.
    spread_all_std_i : Std_i Var_{j,m}(X | i).
    spread_ens_std_i : Std_i Var_j(E_{m}[X | i]).
    spread_models_std_i : Std_i Var_m(E_{j}[X | i]).
    interaction_std_i : Std_i interaction term between ensemble and model.

    mean_true : Mean of truth over (init).
    x_min_true : Min of truth over (init).
    x_max_true : Max of truth over (init).
    data_range_true : Data range of truth over (init).
    median_true : Median of truth over (init).
    q25_true : 0.25 quantile of truth over (init).
    q75_true : 0.75 quantile of truth over (init).
    iqr_true : Interquartile range of truth over (init).

    mean : Mean of all runs over (init, ens, model).
    x_min : Min of all runs over (init, ens, model).
    x_max : Max of all runs over (init, ens, model).
    data_range : Max−min over (init, ens, model).
    median : Median of all runs over (init, ens, model).
    q25 : 0.25 quantile over (init, ens, model).
    q75 : 0.75 quantile over (init, ens, model).
    iqr : Interquartile range over (init, ens, model).
    data_range_mean_over_i : Mean_i of per-initial-state data range.
    iqr_mean_over_i : Mean_i of per-initial-state IQR.
    data_range_std_over_i : Std_i of per-initial-state data range.
    iqr_std_over_i : Std_i of per-initial-state IQR.

    rmse_t : RMSE between ensemble mean and truth (computed like Mansfield and Christensen 2025).
    rmse_t_daan : RMSE between ensemble mean and truth (compute like Crommelin and Vanden-Eijnedn 2008).
    ancr_t : Anomaly correlation between ensemble mean and truth.
    diff_itk : Difference between ensemble mean and truth, shape [N_init, T, K].
    std_itk : Standard deviation over members and models, shape [N_init, T, K].
    """
    x_arr = to_numpy(x_arr)
    x_arr_perfect = to_numpy(x_arr_perfect)
    x_true = to_numpy(x_true)
    assert x_arr.ndim == 5, (
        f"Expected x_arr to have shape [N_init, N_ens, N_models, T, K], got {x_arr.shape}"
    )

    metrics = {}

    # ---- Variability of invariant measure ----
    var_perfect = np.var(x_arr_perfect, axis=(0))  # [T, K]
    var_true = np.var(x_true, axis=(0))  # [T, K]
    metrics["var_perfect"] = var_perfect
    metrics["var_true"] = var_true

    # ---- Disentangle spread into components ----
    spread_total = np.var(x_arr, axis=(0, 1, 2))  # [T, K]
    metrics["spread_total"] = spread_total

    spread_jm = np.var(x_arr, axis=(1, 2))  # [N_init, T, K]
    spread_all_avg_i = np.mean(spread_jm, axis=0)  # [T, K]
    spread_all_std_i = np.std(spread_jm, axis=0)  # [T, K]
    metrics["spread_all_avg_i"] = spread_all_avg_i
    metrics["spread_all_std_i"] = spread_all_std_i

    mean_m = np.mean(x_arr, axis=2)  # [N_init, N_ens, T, K]
    mean_j = np.mean(x_arr, axis=1)  # [N_init, N_models, T, K]

    var_j = np.var(mean_m, axis=1)  # [N_init, T, K]
    spread_ens_avg_i = np.mean(var_j, axis=0)  # [T, K]
    spread_ens_std_i = np.std(var_j, axis=0)  # [T, K]
    metrics["spread_ens_avg_i"] = spread_ens_avg_i
    metrics["spread_ens_std_i"] = spread_ens_std_i

    var_m = np.var(mean_j, axis=1)  # [N_init, T, K]
    spread_models_avg_i = np.mean(var_m, axis=0)  # [T, K]
    spread_models_std_i = np.std(var_m, axis=0)  # [T, K]
    metrics["spread_models_avg_i"] = spread_models_avg_i
    metrics["spread_models_std_i"] = spread_models_std_i

    var_jm = np.var(x_arr, axis=(1, 2))  # [N_init, T, K]

    interaction_avg_i = np.mean(var_jm - var_j - var_m, axis=0)  # [T, K]
    interaction_std_i = np.std(var_jm - var_j - var_m, axis=0)  # [T, K]
    metrics["interaction_avg_i"] = interaction_avg_i
    metrics["interaction_std_i"] = interaction_std_i

    sum_components_avg_i = spread_ens_avg_i + spread_models_avg_i + interaction_avg_i
    metrics["sum_components_avg_i"] = sum_components_avg_i
    del (
        spread_jm,
        mean_m,
        mean_j,
        var_jm,
        var_j,
        var_m,
    )

    # ---- Data range ----
    # True data
    mean_true = np.mean(x_true, axis=0)  # [T, K]
    x_min_true = np.min(x_true, axis=0)  # [T, K]
    x_max_true = np.max(x_true, axis=0)  # [T, K]
    data_range_true = x_max_true - x_min_true  # [T, K]
    metrics["mean_true"] = mean_true
    metrics["x_min_true"] = x_min_true
    metrics["x_max_true"] = x_max_true
    metrics["data_range_true"] = data_range_true

    median_true = np.median(x_true, axis=0)  # [T, K]
    q25_true = np.quantile(x_true, 0.25, axis=0)  # [T, K]
    q75_true = np.quantile(x_true, 0.75, axis=0)  # [T, K
    iqr_true = q75_true - q25_true  # [T, K]
    metrics["median_true"] = median_true
    metrics["q25_true"] = q25_true
    metrics["q75_true"] = q75_true
    metrics["iqr_true"] = iqr_true

    # N_init x N_ens x N_models runs data
    # total
    mean_x = np.mean(x_arr, axis=(0, 1, 2))  # [T, K]
    x_min = np.min(x_arr, axis=(0, 1, 2))  # [T, K]
    x_max = np.max(x_arr, axis=(0, 1, 2))  # [T, K]
    data_range = x_max - x_min  # [T, K]
    metrics["mean"] = mean_x
    metrics["x_min"] = x_min
    metrics["x_max"] = x_max
    metrics["data_range"] = data_range

    median = np.median(x_arr, axis=(0, 1, 2))  # [T, K]
    q25 = np.quantile(x_arr, 0.25, axis=(0, 1, 2))  # [T, K]
    q75 = np.quantile(x_arr, 0.75, axis=(0, 1, 2))  # [T, K]
    iqr = q75 - q25  # [T, K]
    metrics["median"] = median
    metrics["q25"] = q25
    metrics["q75"] = q75
    metrics["iqr"] = iqr

    # initial state averaged
    x_min_i = np.min(x_arr, axis=(1, 2))  # (N_init, T, K)
    x_max_i = np.max(x_arr, axis=(1, 2))  # (N_init, T, K)
    data_range_mean_over_i = np.mean(x_max_i - x_min_i, axis=0)  # (T, K)
    data_range_std_over_i = np.std(x_max_i - x_min_i, axis=0)  # (T, K)
    metrics["data_range_mean_over_i"] = data_range_mean_over_i
    metrics["data_range_std_over_i"] = data_range_std_over_i

    q25_i = np.quantile(x_arr, 0.25, axis=(1, 2))  # (N_init, T, K)
    q75_i = np.quantile(x_arr, 0.75, axis=(1, 2))  # (N_init, T, K)
    iqr_mean_over_i = np.mean(q75_i - q25_i, axis=0)  # (T, K)
    iqr_std_over_i = np.std(q75_i - q25_i, axis=0)  # (T, K)
    metrics["iqr_mean_over_i"] = iqr_mean_over_i
    metrics["iqr_std_over_i"] = iqr_std_over_i

    del x_min, x_max, x_min_i, x_max_i, q25_i, q75_i

    # ---- RMSE ----
    # vector metric
    mean_jm = np.mean(x_arr, axis=(1, 2))  # [N_init, T, K]
    rmse_t = np.sqrt(np.mean((mean_jm - x_true) ** 2, axis=(0, 2)))  # [T]
    metrics["rmse_t"] = rmse_t

    reduce_k = np.sum((mean_jm - x_true) ** 2, axis=-1)  # [N_init, T]
    rmse_daan_t = np.sqrt(np.mean(reduce_k, axis=0))  # [T]
    metrics["rmse_t_daan"] = rmse_daan_t

    # ---- Anomaly correlation ----
    # vector metric
    x_true_mean = np.mean(x_true, axis=(0, 1))  # [K]

    x_true_anom = x_true - x_true_mean[None, None, :]  # [N_init, T, K]
    x_anom = mean_jm - x_true_mean[None, None, :]  # [N_init, T, K]

    num = np.sum(x_true_anom * x_anom, axis=-1)  # [N_init, T]
    den = np.sqrt(
        np.sum(x_true_anom**2, axis=-1) * np.sum(x_anom**2, axis=-1)
    )  # [N_init, T]
    corr_i = num / (den + 1e-12)  # [N_init, T]
    ancr_t = np.mean(corr_i, axis=0)  # [T]
    metrics["ancr_t"] = ancr_t

    del x_true_mean, x_true_anom, x_anom, num, den, corr_i

    # ---- Spread-error relationship ----
    # Difference (RMSE will be computed per bin
    diff_itk = mean_jm - x_true  # [N_init, T, K]
    metrics["diff_itk"] = diff_itk

    # Spread
    std_itk = np.std(x_arr, axis=(1, 2))  # [N_init, T, K]
    metrics["std_itk"] = std_itk

    del std_itk, diff_itk, mean_jm
    return metrics


def validate_config_for_sweep(s, d, N, M, L, setting):
    sweep_name = get_sweep_name(s)
    out_path = d / sweep_name

    # load config
    config_path = out_path / "config.yaml"
    config = ConfigGCM(config_path, eval_mode=True)

    states_type = "perfect" if setting == "perfect" else "perturbed"

    assert config.n_init_states == N, (
        f"Expected n_init_states={N}, got {config.n_init_states} in {config_path}"
    )
    assert config.n_ens_members == M, (
        f"Expected n_ens_members={M}, got {config.n_ens_members} in {config_path}"
    )
    assert config.n_models == L, (
        f"Expected n_models={L}, got {config.n_models} in {config_path}"
    )
    assert config.init_states_type == states_type, (
        f"Expected {states_type} init states, got {config.init_states_type} in {config_path}"
    )
    return config, out_path, sweep_name


def compute_metrics_full_perfect_streamed(gcm_dir, l96_dir, N, M, L):
    """
    N = n_init_states
    M = n_ensemble_members (perturbations of initial states)
    L = n_models (different stochasticity realizations or different coefficients)

    Returns a DataFrame with rows:
      model, config_key, c, f_schedule, noise_type, ar_order, delta_t, truth_config_key
      and computed metrics
    """
    print(f"\nLoading L96 truth from {l96_dir.name}...")
    l96_dir = l96_dir / "short"
    config_full = ConfigL96(l96_dir / "config.yaml", eval_mode=True)
    x_true, t_ref = load_output_l96_ensemble(
        config_full.output_dir(l96_dir), load_y=False, backend=config_full.load_backend
    )
    x_true = x_true[:, 0, ...]  # drop singleton ensemble dimension
    assert x_true.shape == (N, t_ref.size, config_full.K), (
        f"Expected x_true shape {(N, t_ref.size, config_full.K)}, got {x_true.shape}"
    )

    rows = []

    model_name = gcm_dir.name
    print(f"\nProcessing model: {model_name}")

    d_full = gcm_dir / "full"
    d_perfect = gcm_dir / "perfect"
    sweep_path = d_full / "sweep.yaml"
    sweep = load_yaml(sweep_path)

    for s in generate_sweep_dict_list(sweep):
        config_full, out_path_full, sweep_name_full = validate_config_for_sweep(
            s, d_full, N, M, L, setting="perturbed"
        )
        x_gcm, t_curr = load_output_gcm_ensemble(
            config_full.output_dir(out_path_full), backend=config_full.load_backend
        )
        assert np.allclose(t_ref, t_curr), (
            "Time arrays do not match across experiments."
        )

        # Also load corresponding perfect setting model runs here
        config_perfect, out_path_perfect, sweep_name_perfect = (
            validate_config_for_sweep(s, d_perfect, N, 1, 1, setting="perfect")
        )
        x_gcm_perfect, t_curr_perfect = load_output_gcm_ensemble(
            config_perfect.output_dir(out_path_perfect),
            backend=config_perfect.load_backend,
        )
        assert np.allclose(t_ref, t_curr_perfect), (
            "Time arrays do not match across experiments."
        )
        x_gcm_perfect = x_gcm_perfect[
            :, 0, 0, ...
        ]  # drop singleton ensemble and model dimension
        assert x_gcm_perfect.shape == (N, t_ref.size, config_full.K), (
            f"Expected x_gcm_perfect shape {(N, t_ref.size, config_full.K)}, got {x_gcm_perfect.shape}"
        )

        metrics = compute_full_perfect_metrics(x_gcm, x_gcm_perfect, x_true)

        meta = parse_model_meta_from_cfg_key(sweep_name_full, sweep)
        # in case of baseline_ar_p, if ar_order is not in sweep, set to 1 by default
        if "baseline_ar_p" in model_name and meta["ar_order"] is None:
            meta["ar_order"] = 1
        meta["model"] = model_name
        meta["c"] = config_full.c
        meta["si"] = config_full.si
        if "f_schedule" in s:
            f_schedule_key = get_sweep_name({"f_schedule": s["f_schedule"]})
            meta["f_schedule_key"] = f_schedule_key
            meta["truth_config_key"] = f_schedule_key
        else:
            meta["f_schedule"] = forcing_schedule_to_dict(config_full.f_schedule)
            meta["truth_config_key"] = DEFAULT_SWEEP_KEY

        row = {
            "model": model_name,
            **meta,
            **metrics,
        }
        rows.append(row)

        del metrics, x_gcm

    return pd.DataFrame(rows), t_ref
