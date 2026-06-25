"""Train conditional normalizing-flow parameterizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.training.data import TensorLike, make_paired_loader
from parameterization.flow.training.lightning_module import FlowLightningModule

# -----------------------------------------------------------------------------
# Trainer configuration helpers
# -----------------------------------------------------------------------------


def _resolve_monitor_key(monitor: str, has_val: bool) -> str:
    """Map a user-friendly monitor string to the metric key we log."""
    monitor_key = monitor.lower().strip()
    if monitor_key in {"val", "val_loss"}:
        if not has_val:
            raise ValueError("Validation data is required to monitor val loss.")
        return "val_loss"
    if monitor_key in {"train", "train_loss"}:
        return "train_loss"
    raise ValueError("early_stopping_monitor must be 'train' or 'val'.")


DevicesArg = Union[int, List[int], str]  # e.g. 1, 4, [0,1], "auto"


def _resolve_accelerator(
    device: Optional[Union[str, torch.device]],
    devices: DevicesArg = 1,
) -> Tuple[str, DevicesArg, torch.device]:
    """
    Decide Lightning accelerator/devices and model torch.device.

    - If device is CPU -> accelerator="cpu", devices=1
    - If device is CUDA/GPU -> accelerator="gpu", devices as provided
    - For multi-GPU, return model_device="cpu" (Lightning/DDP will move model per process)
    """
    if device is None:
        # Default: CPU
        return "cpu", 1, torch.device("cpu")

    if isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device

    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available.")

        # If user passed an int > 1, we are in multi-GPU territory (single node).
        multi_gpu = isinstance(devices, int) and devices > 1
        if multi_gpu:
            # Important: don't pin the model to cuda:0; Lightning/DDP will handle placement.
            return "gpu", devices, torch.device("cpu")

        # Single GPU case: we can place model on cuda:0 (or just cpu; either works).
        return "gpu", devices, torch.device("cuda:0")

    # CPU fallback (also covers "mps" if you ever add it later)
    return "cpu", 1, torch.device("cpu")


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------


class LossHistory(pl.Callback):
    """Store per-epoch train/val losses in Python lists."""

    def __init__(self) -> None:
        super().__init__()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            self.train_losses.append(float(train_loss.detach().cpu()))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(float(val_loss.detach().cpu()))


class LossPrinter(pl.Callback):
    """Print losses every `log_every` epochs (works with or without validation)."""

    def __init__(self, log_every: int) -> None:
        super().__init__()
        self.log_every = int(log_every)

    def _should_print(self, trainer) -> bool:
        if self.log_every <= 0:
            return False
        epoch = trainer.current_epoch + 1
        return (epoch % self.log_every) == 0

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._should_print(trainer):
            return
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        if train_loss is not None and val_loss is not None:
            epoch = trainer.current_epoch + 1
            print(
                f"epoch {epoch}: train_loss={float(train_loss):.6f} "
                f"val_loss={float(val_loss):.6f}"
            )

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        # If validation is present, printing happens in on_validation_epoch_end.
        if "val_loss" in trainer.callback_metrics:
            return
        if not self._should_print(trainer):
            return
        train_loss = trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            epoch = trainer.current_epoch + 1
            print(f"epoch {epoch}: train_loss={float(train_loss):.6f}")


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def train_conditional_realnvp(
    model: ConditionalRealNVP,
    x_train: TensorLike,
    u_train: TensorLike,
    x_val: Optional[TensorLike],
    u_val: Optional[TensorLike],
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F_train: Optional[TensorLike] = None,
    F_val: Optional[TensorLike] = None,
    seq_len: int = 32,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = None,
    shuffle: bool = True,
    log_every: int = 10,
    log_every_n_batches: int = 50,
    tensorboard_log_dir: Optional[Union[str, Path]] = None,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    early_stopping_monitor: str = "val",
    checkpoint_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    devices: Union[int, List[int], str] = 1,
    strategy: Optional[str] = None,
    num_workers: int = 1,
) -> Tuple[ConditionalRealNVP, List[float], Dict[str, Any]]:
    """
    Train a ConditionalRealNVP on time-series data using PyTorch Lightning.

    The conditioning history is built inside the DataLoader. For each time t,
    the condition includes x at time t and at the previous delta_t full time
    steps, and optionally the forcing at time t.

    Args:
        model: ConditionalRealNVP instance to train. Updated in place.
        x_train: Raw time-series input x(t), shape (N,) or (N, x_dim).
        u_train: Target time-series u(t), shape (N,) or (N, u_dim).
        x_val: Optional validation time-series input x(t), same format as x_train.
        u_val: Optional validation targets u(t), same format as u_train.
        delta_t: Number of full time steps to include in the conditioning history.
        si: Sampling interval of the data.
        dt_full: Physical length of one full time step.
        F_train: Optional scalar forcing time-series aligned with x_train.
        F_val: Optional scalar forcing time-series aligned with x_val.
        seq_len: Length of each contiguous training window.
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        lr: Learning rate for the optimizer.
        weight_decay: Weight decay for the optimizer.
        grad_clip: Optional gradient clipping value.
        shuffle: Whether to shuffle training data.
        log_every: Print losses every this many epochs.
        log_every_n_batches: Logging frequency in batches.
        tensorboard_log_dir: Optional directory for TensorBoard logs.
        early_stopping_patience: Number of epochs with no improvement before stopping.
        early_stopping_min_delta: Minimum improvement required to reset early stopping.
        early_stopping_monitor: Metric to monitor, either "train" or "val".
        checkpoint_dir: Directory for saving model checkpoints.
        device: Torch device for the model.
        devices: Devices argument passed to PyTorch Lightning.
        strategy: Distributed training strategy.
        num_workers: Number of DataLoader workers.

    Returns:
        model: The trained ConditionalRealNVP.
        train_losses: List of per-epoch training losses.
        training_state: Dictionary with training metadata and best checkpoint info.
    """
    accelerator, lightning_devices, model_device = _resolve_accelerator(device, devices)
    model = model.to(model_device)

    # If the base distribution is AR(p), ensure seq_len is long enough
    p = getattr(model.base, "p", 0)
    assert seq_len > p, (
        f"seq_len must be > p for AR(p) base. Got seq_len={seq_len}, p={p}."
    )

    if strategy is None:
        if (
            accelerator == "gpu"
            and isinstance(lightning_devices, int)
            and lightning_devices > 1
        ):
            strategy = "ddp"

    train_loader = make_paired_loader(
        x_train,
        u_train,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(accelerator == "gpu"),
        num_workers=num_workers,
        delta_t=delta_t,
        si=si,
        dt_full=dt_full,
        F=F_train,
        seq_len=seq_len,
    )

    val_loader = None
    if x_val is not None and u_val is not None:
        val_loader = make_paired_loader(
            x_val,
            u_val,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=(accelerator == "gpu"),
            num_workers=num_workers,
            delta_t=delta_t,
            si=si,
            dt_full=dt_full,
            F=F_val,
            seq_len=seq_len,
        )

    monitor_key = _resolve_monitor_key(
        early_stopping_monitor, has_val=(val_loader is not None)
    )

    callbacks: List[pl.Callback] = []
    loss_history = LossHistory()
    callbacks.append(loss_history)

    if log_every:
        callbacks.append(LossPrinter(log_every=log_every))

    early_stopping_callback = None
    if early_stopping_patience > 0:
        early_stopping_callback = EarlyStopping(
            monitor=monitor_key,
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
            check_on_train_epoch_end=(monitor_key == "train_loss"),
        )
        callbacks.append(early_stopping_callback)

    if checkpoint_dir is None:
        checkpoint_dir = (
            Path(tensorboard_log_dir) / "checkpoints"
            if tensorboard_log_dir
            else Path("checkpoints")
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        monitor=monitor_key,
        mode="min",
        save_top_k=1,
        save_last=False,
        save_on_train_epoch_end=(monitor_key == "train_loss"),
    )
    callbacks.append(checkpoint_callback)

    logger = (
        TensorBoardLogger(save_dir=str(tensorboard_log_dir), name="flow")
        if tensorboard_log_dir
        else None
    )

    # Build trainer
    trainer_kwargs = dict(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=lightning_devices,
        logger=logger if logger is not None else False,
        callbacks=callbacks,
        gradient_clip_val=float(grad_clip) if grad_clip else 0.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=log_every_n_batches if log_every_n_batches > 0 else 50,
    )

    if strategy is not None:
        trainer_kwargs["strategy"] = strategy

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    pl_module = FlowLightningModule(model=model, lr=lr, weight_decay=weight_decay)
    trainer.fit(pl_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    train_losses = loss_history.train_losses
    val_losses = loss_history.val_losses

    best_score = checkpoint_callback.best_model_score
    best_loss = float(best_score) if best_score is not None else float("inf")

    stopped_early = bool(
        early_stopping_callback and early_stopping_callback.stopped_epoch > 0
    )

    if monitor_key == "val_loss" and val_losses:
        best_epoch = int(torch.tensor(val_losses).argmin().item())
    elif monitor_key == "train_loss" and train_losses:
        best_epoch = int(torch.tensor(train_losses).argmin().item())
    else:
        best_epoch = 0

    training_state: Dict[str, Any] = {
        "epochs_trained": len(train_losses),
        "best_epoch": best_epoch,
        "best_loss": best_loss,
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "early_stopping_monitor": monitor_key,
        "tensorboard_log_dir": logger.log_dir if logger is not None else None,
        "best_model_path": checkpoint_callback.best_model_path,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    return model, train_losses, training_state


def evaluate_conditional_realnvp(
    model: ConditionalRealNVP,
    x_test: TensorLike,
    u_test: TensorLike,
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F_test: Optional[TensorLike] = None,
    seq_len: int = 32,
    batch_size: int = 256,
    tensorboard_log_dir: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate a trained ConditionalRealNVP on test data.

    The conditioning history is built in the same way as during training.
    For each time t, the condition includes x at time t and at the previous
    delta_t full time steps, and optionally the forcing at time t.

    Args:
        model: Trained ConditionalRealNVP to evaluate.
        x_test: Raw time-series input x(t), shape (N,) or (N, x_dim).
        u_test: Target time-series u(t), shape (N,) or (N, u_dim).
        delta_t: Number of full time steps in the conditioning history.
        si: Sampling interval of the data.
        dt_full: Physical length of one full time step.
        F_test: Optional scalar forcing time-series aligned with x_test.
        seq_len: Length of each contiguous training window.
        batch_size: Batch size for evaluation.
        tensorboard_log_dir: Optional directory for TensorBoard logs.
        device: Torch device for the model.
        num_workers: Number of DataLoader workers.

    Returns:
        Dictionary with:
            - "test_loss": mean test loss over the dataset
            - "tensorboard_log_dir": TensorBoard log directory or None
    """
    accelerator, lightning_devices, model_device = _resolve_accelerator(device, 1)
    model = model.to(model_device)

    # If the base distribution is AR(p), ensure seq_len is long enough
    p = getattr(model.base, "p", 0)
    assert seq_len > p, (
        f"seq_len must be > p for AR(p) base. Got seq_len={seq_len}, p={p}."
    )

    test_loader = make_paired_loader(
        x_test,
        u_test,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(accelerator == "gpu"),
        num_workers=num_workers,
        delta_t=delta_t,
        si=si,
        dt_full=dt_full,
        F=F_test,
        seq_len=seq_len,
    )

    logger = (
        TensorBoardLogger(save_dir=str(tensorboard_log_dir), name="flow_test")
        if tensorboard_log_dir
        else None
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=lightning_devices,
        logger=logger if logger is not None else False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    pl_module = FlowLightningModule(model=model, lr=1e-3, weight_decay=0.0)
    results = trainer.test(pl_module, dataloaders=test_loader, verbose=False)

    test_loss = results[0].get("test_loss") if results else None
    if test_loss is not None:
        print(f"test: loss={float(test_loss):.6f}")

    return {
        "test_loss": float(test_loss) if test_loss is not None else None,
        "tensorboard_log_dir": logger.log_dir if logger is not None else None,
    }
