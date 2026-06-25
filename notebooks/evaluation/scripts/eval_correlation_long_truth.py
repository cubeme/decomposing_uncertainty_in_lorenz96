import numpy as np
import pandas as pd
from evaluation.scripts.script_utils import (
    load_yaml,
    parse_model_meta_from_cfg_key,
)
from notebook_utils import generate_sweep_dict_list

from ensemble.storage import load_output_gcm_ensemble, load_output_l96_ensemble
from models.forcing_schedule import forcing_schedule_to_dict
from utils.config import ConfigGCM, ConfigL96
from utils.sweep_utils import get_sweep_name

DEFAULT_SWEEP_KEY = "default"


def compute_correlation_ensemble_per_k(x_arr, max_lag=None):
    """
    Compute autocorrelation and neighbor cross-correlation functions for
    single spatial index k for array of shape [T, K].
    """
    T, K = x_arr.shape
    if max_lag is None:
        max_lag = T - 1

    out_acf = np.zeros((max_lag + 1, K))
    out_ccf = np.zeros((max_lag + 1, K))

    for k in range(K):
        x_k = x_arr[:, k]  # shape [T]
        x_k_plus_1 = x_arr[:, (k + 1) % K]  # shape [T]
        x_k_centered = x_k - x_k.mean()
        x_k_plus_1_centered = x_k_plus_1 - x_k_plus_1.mean()

        # scalar variances
        var_x_k = np.mean(x_k_centered**2)
        var_x_k_plus_1 = np.mean(x_k_plus_1_centered**2)

        auto_corr = np.array(
            [
                np.mean(x_k_centered[: T - tau] * x_k_centered[tau:])
                for tau in range(max_lag + 1)
            ]
        )  # shape [tau]

        cross_corr = np.array(
            [
                np.mean(x_k_centered[: T - tau] * x_k_plus_1_centered[tau:])
                for tau in range(max_lag + 1)
            ]
        )  # shape [tau]

        out_acf[:, k] = auto_corr / var_x_k
        out_ccf[:, k] = cross_corr / np.sqrt(var_x_k * var_x_k_plus_1)

    return {"acf": out_acf, "ccf": out_ccf}


def compute_correlation_ensemble_per_k_fft(x_arr, max_lag=None):
    """
    Compute autocorrelation and neighbor cross-correlation functions for
    single spatial index k for array of shape [T, K].
    This should be faster than the more intuitive implementation in compute_correlation_ensemble_pooled.

    Returns:
        acf, ccf: [max_lag+1, K]
    """
    T, K = x_arr.shape
    if max_lag is None:
        max_lag = T - 1

    # center per trajectory
    x = x_arr - x_arr.mean(axis=0, keepdims=True)  # shape [T, K]

    # zero-padding for linear correlation
    nfft = 2 * T

    # FFT along time
    Xf = np.fft.rfft(x, n=nfft, axis=0)  # [freq, K]

    # autocovariance
    Sxx = Xf * np.conj(Xf)
    acov = np.fft.irfft(Sxx, n=nfft, axis=0)[:T, :]  # [T, K]

    # cross-covariance with neighbor
    Xf_shift = np.roll(Xf, shift=-1, axis=1)
    Sxy = Xf * np.conj(Xf_shift)
    ccov = np.fft.irfft(Sxy, n=nfft, axis=0)[:T, :]  # [T, K]

    # normalize by (T - tau)
    norm = np.arange(T, 0, -1)[:, None]  # [T, 1]
    acov /= norm
    ccov /= norm

    # variances (per spatial index)
    var = np.mean(x**2, axis=0)  # [K]
    var_shift = np.roll(var, -1)  # neighbor variances

    # normalize
    acf = acov[: max_lag + 1] / var[None, :]
    ccf = ccov[: max_lag + 1] / np.sqrt(var[None, :] * var_shift[None, :])

    return {"acf": acf, "ccf": ccf}


def validate_config_for_sweep(s, d):
    sweep_name = get_sweep_name(s)
    out_path = d / sweep_name

    # load config
    config_path = out_path / "config.yaml"
    config = ConfigGCM(config_path, eval_mode=True)

    assert config.n_ens_members == 1, (
        f"Expected n_ens_members=1, got {config.n_ens_members} in {config_path}"
    )
    assert config.n_models == 1, (
        f"Expected n_models=1, got {config.n_models} in {config_path}"
    )
    assert config.init_states_type == "perfect", (
        f"Expected perfect init states, got {config.init_states_type} in {config_path}"
    )
    return config, out_path, sweep_name


def compute_correlation_long_streamed(gcm_dir, max_lag=None):
    """
    Compute autocorrelation and neighbor cross-correlation functions for reduced models
    from one long integration from perfect initial state.

    Returns a DataFrame with rows:
      model, config_key, c, f_schedule, noise_type, ar_order, delta_t, truth_config_key
      and computed metrics
    """
    rows = []

    model_name = gcm_dir.name
    print(f"\nProcessing model: {model_name}")

    d = gcm_dir / "long"
    sweep_path = d / "sweep.yaml"
    sweep = load_yaml(sweep_path)

    for s in generate_sweep_dict_list(sweep):
        config_full, out_path_full, sweep_name_full = validate_config_for_sweep(s, d)
        x_gcm, t = load_output_gcm_ensemble(
            config_full.output_dir(out_path_full), backend=config_full.load_backend
        )
        print(f"Loaded data with shape {x_gcm.shape}")
        # select first (and only) ensemble member and model, shape [T, K]
        x_gcm = x_gcm[0, 0, 0, ...]
        assert x_gcm.ndim == 2, (
            f"Expected x_gcm to have shape [T, K], got {x_gcm.shape}"
        )

        metrics = compute_correlation_ensemble_per_k_fft(x_gcm, max_lag=max_lag)

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

    return pd.DataFrame(rows), t


def compute_correlation_truth_streamed(l96_dir, max_lag=None):
    """
    Compute autocorrelation and neighbor cross-correlation functions for fully resolved model
    from one long integration from perfect initial state.

    Returns a DataFrame with rows:

      model, config_key, c, f_schedule, noise_type, ar_order, delta_t, truth_config_key
      and computed metrics
    """
    print(f"\nLoading L96 truth from {l96_dir.name}...")
    l96_dir = l96_dir / "long"
    config_l96 = ConfigL96(l96_dir / "config.yaml", eval_mode=True)
    x_true, t = load_output_l96_ensemble(
        config_l96.output_dir(l96_dir), load_y=False, backend=config_l96.load_backend
    )
    # select first (and only) ensemble member, shape [T, K]
    x_true = x_true[0, 0, ...]
    assert x_true.ndim == 2, f"Expected x_true to have shape [T, K], got {x_true.shape}"

    metrics = compute_correlation_ensemble_per_k_fft(x_true, max_lag=max_lag)

    row = {
        "model": l96_dir.parent.name,
        "config_key": DEFAULT_SWEEP_KEY,
        "truth_config_key": DEFAULT_SWEEP_KEY,
        "c": config_l96.c,
        "si": config_l96.si,
        "f_schedule": forcing_schedule_to_dict(config_l96.f_schedule),
        **metrics,
    }

    return pd.DataFrame([row]), t
