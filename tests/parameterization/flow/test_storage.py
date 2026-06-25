import torch

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.storage import load_checkpoint, save_checkpoint


def _assert_optimizer_state_equal(state, expected_state):
    assert state["param_groups"] and expected_state["param_groups"]
    assert len(state["param_groups"]) == len(expected_state["param_groups"])

    for group, expected_group in zip(
        state["param_groups"], expected_state["param_groups"]
    ):
        group_meta = {k: v for k, v in group.items() if k != "params"}
        expected_meta = {k: v for k, v in expected_group.items() if k != "params"}
        assert group_meta == expected_meta

    state_values = [state["state"][k] for k in sorted(state["state"].keys())]
    expected_values = [
        expected_state["state"][k] for k in sorted(expected_state["state"].keys())
    ]
    assert len(state_values) == len(expected_values)

    for value, expected_value in zip(state_values, expected_values):
        assert value.keys() == expected_value.keys()
        for key in value:
            torch.testing.assert_close(value[key], expected_value[key])


def test_save_load_checkpoint_with_model_and_optimizer(temp_dir):
    torch.manual_seed(0)
    cfg = {
        "dim": 4,
        "cond_dim": 3,
        "n_coupling_layers": 2,
        "hidden_dims": (8, 8),
    }
    model = ConditionalRealNVP(**cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    u = torch.randn(5, 2, cfg["dim"])
    cond = torch.randn(5, 2, cfg["cond_dim"])
    loss = -model.log_prob_seq(u, cond).mean()
    loss.backward()
    optimizer.step()

    training_state = {"loss": float(loss.item())}
    save_checkpoint(
        temp_dir,
        model,
        optimizer=optimizer,
        cfg=cfg,
        training_state=training_state,
        epoch=2,
        global_step=10,
        best_metric=0.5,
        include_rng_state=True,
    )

    new_model = ConditionalRealNVP(**cfg)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    loaded_model, checkpoint = load_checkpoint(
        temp_dir, model=new_model, optimizer=new_optimizer, load_rng_state=True
    )

    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded_model.state_dict()[key])

    _assert_optimizer_state_equal(
        new_optimizer.state_dict(), checkpoint["optimizer_state_dict"]
    )
    assert checkpoint["training_state"] == training_state
    assert checkpoint["epoch"] == 2
    assert checkpoint["global_step"] == 10
    assert checkpoint["best_metric"] == 0.5


def test_save_load_checkpoint_with_model_cls(temp_dir):
    torch.manual_seed(1)
    model = ConditionalRealNVP(
        dim=3, cond_dim=2, n_coupling_layers=2, hidden_dims=(4, 4)
    )
    cfg = model.get_config()
    save_checkpoint(temp_dir, model, cfg=cfg)

    loaded_model, checkpoint = load_checkpoint(temp_dir, model=None)

    assert isinstance(loaded_model, ConditionalRealNVP)
    assert loaded_model.get_config() == cfg
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded_model.state_dict()[key])
    assert checkpoint["cfg"] == cfg


def test_save_load_checkpoint_with_arp_base(temp_dir):
    torch.manual_seed(2)
    base = ARpBase(dim=3, p=2, init_rho=[0.2, -0.1], init_sigma=0.7)
    model = ConditionalRealNVP(
        dim=3, cond_dim=2, n_coupling_layers=2, hidden_dims=(4, 4), base_dist=base
    )

    cfg = model.get_config()

    save_checkpoint(temp_dir, model, cfg=cfg)

    loaded_model, _ = load_checkpoint(temp_dir, model=None)

    assert isinstance(loaded_model.base, ARpBase)
    assert loaded_model.base.p == 2
    torch.testing.assert_close(
        loaded_model.base.raw_kappa.detach(),
        model.base.raw_kappa.detach(),
    )
    torch.testing.assert_close(
        loaded_model.base.log_sigma.detach(),
        model.base.log_sigma.detach(),
    )
