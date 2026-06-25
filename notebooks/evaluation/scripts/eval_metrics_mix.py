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


def compute_mix_metrics(x_arr, x_true):
    """
    x_arr : [N_init, N_ens, T, K]
    x_true : [N_init, T, K]

    Compute metrics for single spatial index k.

    Returns dict of arrays (all shape [T, K]):

    spread_total : Var over (init, ens, k).
    spread_all_avg_i : Mean_i Var_{j,k}(X | i).
    spread_all_std_i : Std_i Var_{j,k}(X | i).

    mean : Mean of all runs over (init, ens, model, k).
    x_min : Min of all runs over (init, ens, model, k).
    x_max : Max of all runs over (init, ens, model, k).
    data_range : Max−min over (init, ens, k).
    median : Median of all runs over (init, ens, model, k).
    q25 : 0.25 quantile over (init, ens, k).
    q75 : 0.75 quantile over (init, ens, k).
    iqr : Interquartile range over (init, ens, k).
    data_range_mean_over_i : Mean_i of per-initial-state data range.
    data_range_std_over_i : Std_i of per-initial-state data range.
    iqr_mean_over_i : Mean_i of per-initial-state IQR.
    iqr_std_over_i : Std_i of per-initial-state IQR.

    rmse_t : RMSE between ensemble mean and truth (computed like Mansfield and Christensen 2025).
    rmse_t_daan : RMSE between ensemble mean and truth (compute like Crommelin and Vanden-Eijnedn 2008).
    ancr_t : Anomaly correlation between ensemble mean and truth.
    diff_itk : Difference between ensemble mean and truth, shape [N_init, T, K].
    std_itk : Standard deviation over members, shape [N_init, T, K].
    """
    x_arr = to_numpy(x_arr)
    assert x_arr.ndim == 4, (
        f"Expected x_arr to have shape [N_init, N_ens, T, K], got {x_arr.shape}"
    )

    metrics = {}

    # ---- Disentangle spread into components ----
    spread_total = np.var(x_arr, axis=(0, 1))  # [T, K]
    metrics["spread_total"] = spread_total

    spread_j = np.var(x_arr, axis=1)  # [N_init, T, K]
    spread_all_avg_i = np.mean(spread_j, axis=0)  # [T, K]
    spread_all_std_i = np.std(spread_j, axis=0)  # [T, K]
    metrics["spread_all_avg_i"] = spread_all_avg_i
    metrics["spread_all_std_i"] = spread_all_std_i

    del spread_j

    # ---- Data range ----
    mean_x = np.mean(x_arr, axis=(0, 1))  # [T, K]
    x_min = np.min(x_arr, axis=(0, 1))  # [T, K]
    x_max = np.max(x_arr, axis=(0, 1))  # [T, K]
    data_range = x_max - x_min  # [T, K]
    metrics["mean"] = mean_x
    metrics["x_min"] = x_min
    metrics["x_max"] = x_max
    metrics["data_range"] = data_range

    median = np.median(x_arr, axis=(0, 1))  # [T, K]
    q25 = np.quantile(x_arr, 0.25, axis=(0, 1))  # [T, K]
    q75 = np.quantile(x_arr, 0.75, axis=(0, 1))  # [T, K]
    iqr = q75 - q25  # [T, K]
    metrics["median"] = median
    metrics["q25"] = q25
    metrics["q75"] = q75
    metrics["iqr"] = iqr

    x_min_i = np.min(x_arr, axis=1)  # (N_init, T, K)
    x_max_i = np.max(x_arr, axis=1)  # (N_init, T, K)
    data_range_mean_over_i = np.mean(x_max_i - x_min_i, axis=0)  # (T, K)
    data_range_std_over_i = np.std(x_max_i - x_min_i, axis=0)  # (T, K)
    metrics["data_range_mean_over_i"] = data_range_mean_over_i
    metrics["data_range_std_over_i"] = data_range_std_over_i

    q25_i = np.quantile(x_arr, 0.25, axis=1)  # (N_init, T, K)
    q75_i = np.quantile(x_arr, 0.75, axis=1)  # (N_init, T, K)
    iqr_mean_over_i = np.mean(q75_i - q25_i, axis=0)  # (T, K)
    iqr_std_over_i = np.std(q75_i - q25_i, axis=0)  # (T, K)
    metrics["iqr_mean_over_i"] = iqr_mean_over_i
    metrics["iqr_std_over_i"] = iqr_std_over_i

    del x_min, x_max, x_min_i, x_max_i, q25_i, q75_i

    # ---- RMSE ----
    mean_j = np.mean(x_arr, axis=1)  # [N_init, T, K]
    rmse_t = np.sqrt(np.mean((mean_j - x_true) ** 2, axis=(0, 2)))  # [T]
    metrics["rmse_t"] = rmse_t

    reduce_k = np.sum((mean_j - x_true) ** 2, axis=-1)  # [N_init, T]
    rmse_daan_t = np.sqrt(np.mean(reduce_k, axis=0))  # [T]
    metrics["rmse_t_daan"] = rmse_daan_t

    # ---- Anomaly correlation ----
    x_true_mean = np.mean(x_true, axis=(0, 1))  # [K]

    x_true_anom = x_true - x_true_mean[None, None, :]  # [N_init, T, K]
    x_anom = mean_j - x_true_mean[None, None, :]  # [N_init, T, K]

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
    diff_itk = mean_j - x_true  # [N_init, T, K]
    metrics["diff_itk"] = diff_itk

    # Spread
    std_itk = np.std(x_arr, axis=1)  # [N_init, T, K]
    metrics["std_itk"] = std_itk

    del std_itk, diff_itk, mean_j

    return metrics


def compute_metrics_mix_streamed(gcm_dir, l96_dir, N, M):
    """
    N = n_init_states
    M = n_ensemble_members (perturbations of initial states with stochastic physics)

    Returns a DataFrame with rows:
      model, config_key, c, f_schedule, noise_type, ar_order, delta_t, truth_config_key,
      spread_total, spread_all_avg_i, spread_all_std_i, data_range, q25, q75, iqr, data_range_mean_over_i,
      iqr_mean_over_i, data_range_std_over_i, iqr_std_over_i, rmse_t, ancr_t, rms_spread_t
    """
    print(f"\nLoading L96 truth from {l96_dir.name}...")
    l96_dir = l96_dir / "short"
    config = ConfigL96(l96_dir / "config.yaml", eval_mode=True)
    x_true, t_ref = load_output_l96_ensemble(
        config.output_dir(l96_dir), load_y=False, backend=config.load_backend
    )
    x_true = x_true[:, 0, ...]  # drop singleton ensemble dimension
    assert x_true.shape == (N, t_ref.size, config.K), (
        f"Expected x_true shape {(N, t_ref.size, config.K)}, got {x_true.shape}"
    )

    rows = []

    model_name = gcm_dir.name
    print(f"\nProcessing model: {model_name}")

    d = gcm_dir / "mix"
    sweep_path = d / "sweep.yaml"
    sweep = load_yaml(sweep_path)

    for s in generate_sweep_dict_list(sweep):
        sweep_name = get_sweep_name(s)
        out_path = d / sweep_name

        # load config
        config_path = out_path / "config.yaml"
        config = ConfigGCM(config_path, eval_mode=True)

        assert config.n_init_states == N, (
            f"Expected n_init_states={N}, got {config.n_init_states} in {config_path}"
        )
        assert config.n_ens_members == M, (
            f"Expected n_ens_members={M}, got {config.n_ens_members} in {config_path}"
        )
        assert config.n_models == 1, (
            f"Expected n_models=1, got {config.n_models} in {config_path}"
        )
        assert config.init_states_type == "perturbed", (
            f"Expected perturbed init states, got {config.init_states_type} in {config_path}"
        )

        x_gcm, t_curr = load_output_gcm_ensemble(
            config.output_dir(out_path), backend=config.load_backend
        )
        x_gcm = x_gcm[:, :, 0, ...]  # drop singleton model dimension

        assert np.allclose(t_ref, t_curr), (
            "Time arrays do not match across experiments."
        )

        metrics = compute_mix_metrics(x_gcm, x_true)

        meta = parse_model_meta_from_cfg_key(sweep_name, sweep)
        # in case of baseline_ar_p, if ar_order is not in sweep, set to 1 by default
        if "baseline_ar_p" in model_name and meta["ar_order"] is None:
            meta["ar_order"] = 1
        meta["model"] = model_name
        meta["c"] = config.c
        meta["si"] = config.si
        if "f_schedule" in s:
            f_schedule_key = get_sweep_name({"f_schedule": s["f_schedule"]})
            meta["f_schedule_key"] = f_schedule_key
            meta["truth_config_key"] = f_schedule_key
        else:
            meta["f_schedule"] = forcing_schedule_to_dict(config.f_schedule)
            meta["truth_config_key"] = DEFAULT_SWEEP_KEY

        row = {
            "model": model_name,
            **meta,
            **metrics,
        }
        rows.append(row)

        del metrics, x_gcm

    return pd.DataFrame(rows), t_ref
