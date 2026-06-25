from pathlib import Path

import torch

from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.training import (
    evaluate_conditional_realnvp,
    train_conditional_realnvp,
)
from parameterization.flow.training import train as training_module
from parameterization.flow.training.train import _resolve_accelerator


def test_resolve_accelerator_defaults_to_cpu():
    accelerator, devices, model_device = _resolve_accelerator(None, devices=4)

    assert accelerator == "cpu"
    assert devices == 1
    assert model_device.type == "cpu"


def test_resolve_accelerator_cpu_forces_single_device():
    accelerator, devices, model_device = _resolve_accelerator("cpu", devices=4)

    assert accelerator == "cpu"
    assert devices == 1
    assert model_device.type == "cpu"


def test_resolve_accelerator_cuda_single_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    accelerator, devices, model_device = _resolve_accelerator("cuda", devices=1)

    assert accelerator == "gpu"
    assert devices == 1
    assert model_device.type == "cuda"


def test_resolve_accelerator_cuda_multi_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    accelerator, devices, model_device = _resolve_accelerator("cuda", devices=2)

    assert accelerator == "gpu"
    assert devices == 2
    assert model_device.type == "cpu"


def test_train_conditional_realnvp_sets_ddp_strategy_for_multi_gpu(
    monkeypatch, tmp_path
):
    captured = {}

    class DummyTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, *args, **kwargs):
            return None

    monkeypatch.setattr(training_module.pl, "Trainer", DummyTrainer)
    monkeypatch.setattr(
        training_module,
        "_resolve_accelerator",
        lambda device, devices: ("gpu", 2, torch.device("cpu")),
    )

    torch.manual_seed(0)
    x_dim, u_dim = 3, 2
    batch = 6

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x = torch.randn(batch, x_dim)
    u = torch.randn(batch, u_dim)

    train_conditional_realnvp(
        model=model,
        x_train=x,
        u_train=u,
        x_val=None,
        u_val=None,
        epochs=1,
        batch_size=3,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=None,
        shuffle=False,
        log_every=1,
        log_every_n_batches=1,
        tensorboard_log_dir=None,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        early_stopping_monitor="train",
        checkpoint_dir=tmp_path,
        device="cuda",
        devices=2,
        strategy=None,
        num_workers=2,
        seq_len=2,
    )

    assert captured["accelerator"] == "gpu"
    assert captured["devices"] == 2
    assert captured["strategy"] == "ddp"


def test_train_conditional_realnvp_tensorboard_logging(tmp_path):
    torch.manual_seed(0)
    x_dim, u_dim = 3, 2
    batch = 8
    epochs = 2

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x = torch.randn(batch, x_dim)
    u = torch.randn(batch, u_dim)
    x_train, x_val = x[:6], x[6:]
    u_train, u_val = u[:6], u[6:]

    _, _, training_state = train_conditional_realnvp(
        model=model,
        x_train=x_train,
        u_train=u_train,
        x_val=x_val,
        u_val=u_val,
        epochs=epochs,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=None,
        shuffle=False,
        log_every=1,
        log_every_n_batches=1,
        tensorboard_log_dir=tmp_path,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        num_workers=0,
        seq_len=2,
    )

    log_dir = Path(training_state["tensorboard_log_dir"])
    assert log_dir.exists()
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    assert event_files
    assert any(path.stat().st_size > 0 for path in event_files)


def test_train_conditional_realnvp_small_run(tmp_path):
    torch.manual_seed(0)
    x_dim, u_dim = 3, 2
    batch = 12
    epochs = 3

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x = torch.randn(batch, x_dim)
    u = torch.randn(batch, u_dim)
    x_train, x_val = x[:8], x[8:]
    u_train, u_val = u[:8], u[8:]

    trained_model, losses, training_state = train_conditional_realnvp(
        model=model,
        x_train=x_train,
        u_train=u_train,
        x_val=x_val,
        u_val=u_val,
        epochs=epochs,
        batch_size=4,
        lr=1e-3,
        weight_decay=0.0,
        grad_clip=None,
        shuffle=True,
        log_every=1,
        log_every_n_batches=1,
        tensorboard_log_dir=None,
        checkpoint_dir=tmp_path,
        early_stopping_patience=3,
        early_stopping_min_delta=1e-4,
        num_workers=0,
        seq_len=2,
    )

    assert trained_model is model
    assert len(losses) == epochs
    assert training_state["epochs_trained"] == epochs
    assert 1 <= training_state["best_epoch"] <= epochs
    assert torch.isfinite(torch.tensor(losses)).all().item()


def test_evaluate_conditional_realnvp_tensorboard_logging(tmp_path):
    torch.manual_seed(0)
    x_dim, u_dim = 3, 2
    batch = 6

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x = torch.randn(batch, x_dim)
    u = torch.randn(batch, u_dim)

    test_state = evaluate_conditional_realnvp(
        model=model,
        x_test=x,
        u_test=u,
        batch_size=3,
        tensorboard_log_dir=tmp_path,
        num_workers=0,
        seq_len=2,
    )

    assert test_state["test_loss"] is not None
    log_dir = Path(test_state["tensorboard_log_dir"])
    assert log_dir.exists()
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    assert event_files


def test_evaluate_conditional_realnvp_small_run():
    torch.manual_seed(0)
    x_dim, u_dim = 3, 2
    batch = 10

    model = ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )

    x = torch.randn(batch, x_dim)
    u = torch.randn(batch, u_dim)

    test_state = evaluate_conditional_realnvp(
        model=model,
        x_test=x,
        u_test=u,
        batch_size=5,
        tensorboard_log_dir=None,
        num_workers=0,
        seq_len=2,
    )

    assert test_state["test_loss"] is not None
    assert test_state["tensorboard_log_dir"] is None
    assert torch.isfinite(torch.tensor(test_state["test_loss"])).all().item()
