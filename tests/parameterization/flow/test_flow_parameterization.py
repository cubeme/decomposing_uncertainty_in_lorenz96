import pytest
import torch

from parameterization.flow.flow_model import ConditionalRealNVP
from parameterization.flow.flow_parameterization import FlowParameterization


class _CaptureFlow:
    """
    Minimal stand-in for a conditional normalizing flow used in tests.

    This class does not perform any transformation or density evaluation.
    Instead, it records (captures) the most recent conditioning tensor
    passed to `sample` or `log_prob` via the `last_cond` attribute.

    It is used to verify that `FlowParameterization` constructs and passes
    the correct conditioning vector (e.g. state history and forcing)
    without depending on a real flow implementation.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.last_cond = None
        self.cond_dim = None

    def to(self, *_args, **_kwargs):
        return self

    def sample_seq(self, cond: torch.Tensor, z=None) -> torch.Tensor:
        self.last_cond = cond[:, 0, :].detach().cpu()
        self.cond_dim = cond.shape[-1]
        return torch.zeros(
            cond.shape[0], cond.shape[1], self.dim, device=cond.device, dtype=cond.dtype
        )

    def log_prob_seq(self, u: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        self.last_cond = cond[:, 0, :].detach().cpu()
        self.cond_dim = cond.shape[-1]
        return torch.zeros(cond.shape[0], device=cond.device, dtype=cond.dtype)


def _make_flow(x_dim: int, u_dim: int) -> ConditionalRealNVP:
    return ConditionalRealNVP(
        dim=u_dim,
        cond_dim=x_dim,
        n_coupling_layers=2,
        hidden_dims=(8,),
    )


def test_predict_shapes_iid():
    torch.manual_seed(0)
    x_dim, u_dim, batch = 4, 6, 10
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=0,
        rho=0.0,
    )

    x = torch.randn(batch, x_dim)
    u = fp.predict(x)

    assert u.shape == (batch, u_dim)
    assert torch.isfinite(u).all().item()


def test_predict_shapes_iid_vector_squeezed():
    torch.manual_seed(1)
    x_dim, u_dim = 5, 7
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=0,
        rho=0.0,
    )

    x_vec = torch.randn(x_dim)
    u_vec = fp.predict(x_vec)

    assert u_vec.shape == (u_dim,)
    assert torch.isfinite(u_vec).all().item()


def test_log_prob_shapes_and_finite():
    torch.manual_seed(2)
    x_dim, u_dim, batch = 3, 4, 12
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=0,
        rho=0.0,
    )

    x = torch.randn(batch, x_dim)
    # use random u; log prob should still be finite
    u = torch.randn(batch, u_dim)

    lp = fp.log_prob(x, u)
    assert lp.shape == (batch,)
    assert torch.isfinite(lp).all().item()


def test_ar1_latent_update_and_predict():
    torch.manual_seed(3)
    x_dim, u_dim, batch = 4, 5, 8
    rho = 0.7
    sigma = 0.05
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=1,
        rho=rho,
        sigma=sigma,
    )

    x = torch.randn(batch, x_dim)

    # Before predict, latent history is empty
    assert fp._z_hist == []

    # First predict initializes latent state
    u1 = fp.predict(x)
    assert len(fp._z_hist) == 1
    assert fp._z_hist[0].shape == (batch, u_dim)
    assert u1.shape == (batch, u_dim)
    assert torch.isfinite(u1).all().item()

    # Update should change the latent state
    z_before = fp._z_hist[0].clone()
    fp.update()
    z_after = fp._z_hist[0]
    assert torch.isfinite(z_after).all().item()
    # Not guaranteed to always differ for all elements, but very likely
    assert not torch.allclose(z_after, z_before, rtol=1e-6, atol=1e-7)

    # Predict again with updated latent
    u2 = fp.predict(x)
    assert u2.shape == (batch, u_dim)
    assert torch.isfinite(u2).all().item()


def test_to_device_cpu_updates_device_attr():
    torch.manual_seed(4)
    x_dim, u_dim = 3, 5
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=0,
        rho=0.0,
    )

    dev = torch.device("cpu")
    fp.to(dev)
    assert fp.device == dev


def test_predict_uses_history_and_forcing_in_condition():
    x_dim, u_dim = 2, 3
    delta_t = 1
    si = 0.5
    flow = _CaptureFlow(u_dim)
    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=flow,
        ar_order=0,
        rho=0.0,
        delta_t=delta_t,
        si=si,
        include_forcing_in_cond=True,
    )

    fp.predict(torch.tensor([[1.0, 2.0]]), F=10.0)
    fp.predict(torch.tensor([[3.0, 4.0]]), F=20.0)
    fp.predict(torch.tensor([[5.0, 6.0]]), F=30.0)

    cond = flow.last_cond
    assert cond is not None
    assert cond.shape == (1, x_dim * (delta_t + 1) + 1)
    expected = torch.tensor([[5.0, 6.0, 1.0, 2.0, 30.0]])
    torch.testing.assert_close(cond, expected)


def test_arp_requires_sigma_when_p_gt_1():
    torch.manual_seed(5)
    x_dim, u_dim = 3, 4
    rho = [0.3, 0.2]
    with pytest.raises(ValueError, match="sigma must be provided"):
        FlowParameterization(
            x_dim=x_dim,
            u_dim=u_dim,
            flow_model=_make_flow(x_dim, u_dim),
            ar_order=2,
            rho=rho,
        )

    fp = FlowParameterization(
        x_dim=x_dim,
        u_dim=u_dim,
        flow_model=_make_flow(x_dim, u_dim),
        ar_order=2,
        rho=rho,
        sigma=0.5,
    )

    x = torch.randn(6, x_dim)
    u = fp.predict(x)
    assert u.shape == (6, u_dim)
