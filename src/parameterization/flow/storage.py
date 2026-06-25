"""Store and load normalizing-flow checkpoints."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import ConditionalRealNVP


def _capture_rng_state(include_cuda: bool) -> Dict[str, Any]:
    rng_state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
    }

    if include_cuda and torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    else:
        rng_state["cuda"] = None
    return rng_state


def _restore_rng_state(rng_state: Dict[str, Any]) -> None:
    if not rng_state:
        return
    if rng_state.get("python") is not None:
        random.setstate(rng_state["python"])
    if rng_state.get("torch") is not None:
        torch.set_rng_state(rng_state["torch"])
    if rng_state.get("numpy") is not None:
        np.random.set_state(rng_state["numpy"])

    if rng_state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["cuda"])


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    cfg: Optional[Dict[str, Any]] = None,
    training_state: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    global_step: Optional[int] = None,
    best_metric: Optional[float] = None,
    include_rng_state: bool = False,
) -> None:
    # Prepare checkpoint dictionary
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "cfg": cfg if cfg is not None else model.get_config(),
        "training_state": training_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if include_rng_state:
        checkpoint["rng_state"] = _capture_rng_state(include_cuda=True)

    path = Path(output_dir) / "checkpoint.pt"
    torch.save(checkpoint, path)


def load_checkpoint(
    load_dir: Path,
    model: Optional[torch.nn.Module] = None,
    model_cls: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
    load_optimizer: bool = True,
    load_rng_state: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    # Load checkpoint
    checkpoint = torch.load(load_dir / "checkpoint.pt", weights_only=False)

    cfg = checkpoint.get("cfg")
    if not cfg:
        raise ValueError("Checkpoint is missing cfg; provide a model instance.")

    if model is None:
        base_dist = None
        if cfg.get("base_dist_name") == "ar_p":
            base_dist = ARpBase(
                dim=int(cfg["dim"]),
                p=int(cfg["ar_order"]),
                init_rho=cfg.get("init_rho"),
                init_sigma=float(cfg.get("init_sigma", 1.0)),
            )
        if model_cls is None:
            model_cls = ConditionalRealNVP

        model = model_cls(
            dim=int(cfg["dim"]),
            cond_dim=int(cfg["cond_dim"]),
            n_coupling_layers=int(cfg["n_coupling_layers"]),
            hidden_dims=tuple(cfg["hidden_dims"]),
            use_flexible_tails=bool(cfg.get("use_flexible_tails", False)),
            ttf_init_lambda=float(cfg.get("ttf_init_lambda", 0.1)),
            base_dist=base_dist,
        )
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state if provided
    if optimizer is not None and load_optimizer:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            optimizer.load_state_dict(opt_state)

    if load_rng_state:
        _restore_rng_state(checkpoint.get("rng_state", {}))

    return model, checkpoint
