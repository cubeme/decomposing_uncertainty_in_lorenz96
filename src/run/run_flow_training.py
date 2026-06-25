"""Train a conditional normalizing flow on Lorenz '96 data."""

import random
from pathlib import Path

import numpy as np
import torch
from absl import app, flags, logging

from models.forcing_schedule import ConstantForcingSchedule, forcing_at_array
from models.storage import load_output_l96
from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.fit_rho import fit_rho_sigma_p_from_data
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.storage import save_checkpoint
from parameterization.flow.training.train import (
    evaluate_conditional_realnvp,
    train_conditional_realnvp,
)
from parameterization.utils.helpers import compute_coupling_from_x
from parameterization.utils.storage import save_ar_p_parameters
from utils.config import ConfigFlowTraining
from utils.run_helpers import configure_logging
from utils.sweep_utils import keep_only_load_sweep

# ------------------------------- FLAGS ---------------------------------------
FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_string(
        "config",
        None,
        "Path to the YAML configuration file.",
        required=True,
    )


# =============================================================================
# Local functions
# =============================================================================


def _resolve_tensorboard_dir(config, out_dir):
    log_dir = config.tensorboard_log_dir
    if not log_dir:
        return None
    if isinstance(log_dir, bool):
        if log_dir:
            return out_dir / "tensorboard"
        return None
    log_dir = str(log_dir)
    path = Path(log_dir)
    if not path.is_absolute():
        return out_dir / path
    return path


def _save_training_results(
    model,
    config,
    out_dir,
    training_state_extra=None,
):
    output_dir = config.output_dir(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = model.get_config()
    cfg.update(
        {
            "base_dist_name": config.base_dist,
            "ar_order": int(config.ar_order) if config.base_dist == "ar_p" else 0,
            "init_rho": config.init_rho if config.base_dist == "ar_p" else None,
            "init_sigma": float(config.init_sigma)
            if config.base_dist == "ar_p"
            else None,
        }
    )
    training_state = {
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "train_perc": config.train_perc,
        "val_perc": config.val_perc,
        "test_perc": config.test_perc,
        "seq_len": config.seq_len,
        "device": config.device,
        "devices": config.devices,
        "strategy": config.strategy,
        "num_workers_data_loader": config.num_workers_data_loader,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_delta": config.early_stopping_min_delta,
        "early_stopping_monitor": config.early_stopping_monitor,
        "fit_ar_parameters": config.fit_ar_parameters,
        "tensorboard_log_dir": str(config.tensorboard_log_dir)
        if config.tensorboard_log_dir
        else None,
    }

    # Store flow variations when present on config (flow-type specific in your ConfigFlowTraining)
    for key in (
        "use_flexible_tails",
        "ttf_init_lambda",
        "delta_t",
        "include_forcing_in_cond",
        "seq_len",
    ):
        if hasattr(config, key):
            training_state[key] = getattr(config, key)

    if training_state_extra:
        training_state.update(training_state_extra)

    save_checkpoint(output_dir, model=model, cfg=cfg, training_state=training_state)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _fetch_flow_parameters(config):
    """
    Returns (delta_t, include_forcing_in_cond, use_flexible_tails, ttf_init_lambda).
    """
    delta_t = config.delta_t
    include_forcing_in_cond = bool(config.include_forcing_in_cond)
    use_flexible_tails = bool(config.use_flexible_tails)
    ttf_init_lambda = float(config.ttf_init_lambda)
    return delta_t, include_forcing_in_cond, use_flexible_tails, ttf_init_lambda


# =============================================================================
# Run function
# =============================================================================


def run_from_config_path(config_path: str):
    logging.info(f"Loading configuration from {config_path}...")
    config = ConfigFlowTraining(config_path)

    out_dir = config.results_dir / config.experiment_name / config.sweep_name
    configure_logging(out_dir)

    logging.info(f"Set random seed {config.seed}...")
    _set_seed(config.seed)

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------
    logging.info(f"Loading L96 data from {config.l96_data_dir}...")
    x, _, t = load_output_l96(
        config.l96_data_dir
        / keep_only_load_sweep(config.sweep_name, config.load_sweep)
        / config.l96_output_sub_dir,
        backend=config.l96_load_backend,
    )
    if config.l96_load_backend == "zarr":
        x = np.asarray(x)  # loads into memory

    # ---------------------------------------------------------------------------
    # Compute coupling term from X
    # ---------------------------------------------------------------------------
    logging.info("Computing coupling term from X...")
    F_values = forcing_at_array(config.f_schedule, t)  # shape (N,)
    u, x = compute_coupling_from_x(x, config.si, F_values, config.h, config.b, config.c)

    # ---------------------------------------------------------------------------
    # Select training/val/test splits
    # ---------------------------------------------------------------------------
    N = x.shape[0]
    f_type = (
        "constant"
        if isinstance(config.f_schedule, ConstantForcingSchedule)
        else "time-varying"
    )

    n_train = int(config.train_perc * N)
    n_val = int(config.val_perc * N)
    n_test = int(config.test_perc * N)
    n_test = min(n_test, N - n_train - n_val)

    if f_type == "constant":
        # keep your original contiguous splits
        i0 = 0
        i1 = i0 + n_train
        i2 = i1 + n_val
        i3 = i2 + n_test

        idx_train = np.arange(i0, i1)
        idx_val = np.arange(i1, i2)
        idx_test = np.arange(i2, i3)

    else:
        # chunk sampling for linear/oscillating schedules
        chunk_TU = config.chunk_length  # in time units (TU)
        chunk_len = int(round(chunk_TU / config.si))  # indices per chunk
        chunk_len = max(chunk_len, 1)

        n_chunks = N // chunk_len
        if n_chunks < 3:
            raise ValueError(
                f"Not enough data for chunking: N={N}, chunk_len={chunk_len} -> n_chunks={n_chunks}"
            )

        # non-overlapping chunks: [0:chunk_len], [chunk_len:2*chunk_len], ...
        chunks = np.arange(n_chunks)
        rng = np.random.default_rng(config.seed)
        rng.shuffle(chunks)

        n_train_chunks = int(round(config.train_perc * n_chunks))
        n_val_chunks = int(round(config.val_perc * n_chunks))
        n_test_chunks = int(round(config.test_perc * n_chunks))

        # ensure we don't exceed n_chunks
        n_train_chunks = min(n_train_chunks, n_chunks)
        n_val_chunks = min(n_val_chunks, n_chunks - n_train_chunks)
        n_test_chunks = min(n_test_chunks, n_chunks - n_train_chunks - n_val_chunks)

        train_chunks = chunks[:n_train_chunks]
        val_chunks = chunks[n_train_chunks : n_train_chunks + n_val_chunks]
        test_chunks = chunks[
            n_train_chunks + n_val_chunks : n_train_chunks
            + n_val_chunks
            + n_test_chunks
        ]

        def _chunks_to_idx(ch):
            starts = (ch * chunk_len).astype(int)
            idx = np.concatenate([np.arange(s, s + chunk_len) for s in starts])
            return idx[(idx >= 0) & (idx < N)]

        idx_train = _chunks_to_idx(train_chunks)
        idx_val = _chunks_to_idx(val_chunks)
        idx_test = _chunks_to_idx(test_chunks)

    logging.info(
        "Split sizes (time steps): train=%d, val=%d, test=%d (N=%d).",
        idx_train.size,
        idx_val.size,
        idx_test.size,
        N,
    )
    logging.info(
        "Split spans in physical time: train=%.6g, val=%.6g, test=%.6g.",
        idx_train.size * config.si,
        idx_val.size * config.si,
        idx_test.size * config.si,
    )

    x_train, u_train = x[idx_train], u[idx_train]
    x_val, u_val = x[idx_val], u[idx_val]
    x_test, u_test = x[idx_test], u[idx_test]

    F_train = F_values[idx_train]
    F_val = F_values[idx_val]
    F_test = F_values[idx_test]

    # ---------------------------------------------------------------------------
    # Convert data to float32
    # ---------------------------------------------------------------------------
    x_train = x_train.astype("float32", copy=False)
    u_train = u_train.astype("float32", copy=False)
    x_val = x_val.astype("float32", copy=False)
    u_val = u_val.astype("float32", copy=False)

    F_train = F_train.astype("float32", copy=False)
    F_val = F_val.astype("float32", copy=False)

    # ---------------------------------------------------------------------------
    # Flow variations: history, forcing-in-condition, flexible tails
    # ---------------------------------------------------------------------------
    delta_t, include_forcing_in_cond, use_flexible_tails, ttf_init_lambda = (
        _fetch_flow_parameters(config)
    )

    # Compute effective cond_dim given how the loader will construct the condition.
    x_dim = 1 if x_train.ndim == 1 else x_train.shape[1]
    cond_dim = x_dim * (delta_t + 1) + (1 if include_forcing_in_cond else 0)

    if delta_t > 0:
        logging.info(
            "Using history in condition: delta_t=%d.",
            delta_t,
        )
    if include_forcing_in_cond:
        logging.info("Including scalar forcing F_t in condition.")
    if use_flexible_tails:
        logging.info(
            "Using flexible tails (TTF) with ttf_init_lambda=%.6f.", ttf_init_lambda
        )

    # ---------------------------------------------------------------------------
    # Train flow model
    # ---------------------------------------------------------------------------
    hidden_dims = tuple(config.hidden_dims)
    u_dim = 1 if u_train.ndim == 1 else u_train.shape[1]

    logging.info("Initializing ConditionalRealNVP model...")

    base_dist = None
    if config.base_dist == "ar_p":
        base_dist = ARpBase(
            dim=u_dim,
            p=int(config.ar_order),
            init_rho=config.init_rho,
            init_sigma=float(config.init_sigma),
        )

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=cond_dim,
        n_coupling_layers=config.n_coupling_layers,
        hidden_dims=hidden_dims,
        use_flexible_tails=use_flexible_tails,
        ttf_init_lambda=ttf_init_lambda,
        base_dist=base_dist,
    )

    logging.info("Training ConditionalRealNVP...")
    early_stopping_monitor = config.early_stopping_monitor
    if early_stopping_monitor == "val" and x_val.size == 0:
        # This is only the case for val_perc=0.0
        logging.warning("No validation data; falling back to train loss monitoring.")
        early_stopping_monitor = "train"

    tensorboard_log_dir = _resolve_tensorboard_dir(config, out_dir)

    # Decide whether to pass forcing arrays
    F_train_arg = F_train if include_forcing_in_cond else None
    F_val_arg = F_val if include_forcing_in_cond else None

    # Train
    model, _, training_state = train_conditional_realnvp(
        model,
        x_train,
        u_train,
        x_val if x_val.size > 0 else None,
        u_val if u_val.size > 0 else None,
        delta_t=delta_t,
        si=config.si,
        dt_full=config.dt_full,
        F_train=F_train_arg,
        F_val=F_val_arg,
        seq_len=config.seq_len,
        epochs=config.epochs,
        batch_size=config.batch_size,
        lr=config.lr,
        weight_decay=config.weight_decay,
        grad_clip=config.grad_clip,
        tensorboard_log_dir=tensorboard_log_dir,
        checkpoint_dir=out_dir / "checkpoints",
        early_stopping_patience=config.early_stopping_patience,
        early_stopping_min_delta=config.early_stopping_min_delta,
        early_stopping_monitor=early_stopping_monitor,
        device=config.device,
        devices=config.devices,
        strategy=config.strategy,
        num_workers=config.num_workers_data_loader,
    )

    # ---------------------------------------------------------------------------
    # Test set
    # ---------------------------------------------------------------------------
    if config.test_perc > 0.0:
        x_test = x_test.astype("float32", copy=False)
        u_test = u_test.astype("float32", copy=False)
        F_test = F_test.astype("float32", copy=False)

        logging.info(
            "Evaluating trained model on %d time steps...",
            x_test.shape[0],
        )

        eval_results = evaluate_conditional_realnvp(
            model,
            x_test,
            u_test,
            delta_t=delta_t,
            si=config.si,
            dt_full=config.dt_full,
            F_test=(F_test if include_forcing_in_cond else None),
            seq_len=config.seq_len,
            batch_size=config.batch_size,
            tensorboard_log_dir=tensorboard_log_dir,
            device=config.device,
            num_workers=config.num_workers_data_loader,
        )

        training_state.update(
            {
                "test_loss": eval_results.get("test_loss"),
                "test_tensorboard_log_dir": eval_results.get("tensorboard_log_dir"),
            }
        )

    # ---------------------------------------------------------------------------
    # Fit rho if requested
    # ---------------------------------------------------------------------------
    if config.fit_ar_parameters and config.base_dist != "ar_p":
        logging.info(
            "Fitting AR parameters rho and sigma from trained flow with fit method '%s'...",
            config.fit_method,
        )

        ar_orders = (
            config.ar_order if isinstance(config.ar_order, list) else [config.ar_order]
        )

        for ar_order in ar_orders:
            logging.info(
                "Fit parameters AR order p=%d...",
                ar_order,
            )

            rho, sigma = fit_rho_sigma_p_from_data(
                model,
                x_train,
                u_train,
                p=ar_order,
                delta_t=delta_t,
                si=config.si,
                dt_full=config.dt_full,
                F=F_train_arg,
                method=config.fit_method,
            )

            logging.info("Saving AR parameters rho and sigma...")
            save_ar_p_parameters(
                out_path=config.ar_parameters_dir(out_dir),
                rho=rho,
                sigma=sigma,
                ar_order=ar_order,
            )

    elif config.base_dist == "ar_p":
        base = getattr(model, "base", None)
        if base is not None and hasattr(base, "rho") and hasattr(base, "sigma"):
            rho_arr = base.rho.detach().cpu().numpy()
            rho = float(rho_arr.reshape(-1)[0]) if rho_arr.size == 1 else rho_arr
            sigma = float(base.sigma.detach().item())

            logging.info("Saving AR base parameters rho and sigma from trained flow...")
            save_ar_p_parameters(
                out_path=config.ar_parameters_dir(out_dir),
                rho=rho,
                sigma=sigma,
                ar_order=int(getattr(base, "p", config.ar_order)),
            )

    # ---------------------------------------------------------------------------
    # Save model
    # ---------------------------------------------------------------------------
    logging.info("Saving model...")
    _save_training_results(
        model,
        config,
        out_dir,
        training_state_extra=training_state,
    )

    logging.info("DONE")


def main(argv):
    del argv  # unused; absl passes argv list
    run_from_config_path(FLAGS.config)


if __name__ == "__main__":
    define_flags()
    app.run(main)
