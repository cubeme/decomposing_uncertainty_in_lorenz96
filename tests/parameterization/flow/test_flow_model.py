import torch

from parameterization.flow.base_distribution import ARpBase
from parameterization.flow.flow_model import (
    MLP,
    ConditionalAffineCoupling,
    ConditionalRealNVP,
)
from parameterization.flow.tail_transform import TailTransformTTF


def test_mlp_forward_shape_and_finite():
    torch.manual_seed(0)
    in_dim, out_dim = 5, 3
    batch = 10

    mlp = MLP(in_dim=in_dim, out_dim=out_dim, hidden_dims=(16, 16))

    x = torch.randn(batch, in_dim)
    y = mlp(x)

    assert y.shape == (batch, out_dim)
    assert torch.isfinite(y).all().item()


def _make_mask(dim: int):
    # Alternate half-ones, half-zeros mask
    half = dim // 2
    return torch.cat([torch.ones(half), torch.zeros(dim - half)])


def test_conditional_affine_coupling_forward_inverse():
    torch.manual_seed(1)
    dim, cond_dim, batch = 6, 4, 8
    mask = _make_mask(dim)

    layer = ConditionalAffineCoupling(
        dim=dim, cond_dim=cond_dim, mask=mask, hidden_dims=(32, 32)
    )

    x = torch.randn(batch, dim)
    cond = torch.randn(batch, cond_dim)

    # Forward
    y, log_det_fwd = layer(x, cond, inverse=False)
    assert y.shape == (batch, dim)
    assert log_det_fwd.shape == (batch,)
    assert torch.isfinite(y).all().item()
    assert torch.isfinite(log_det_fwd).all().item()

    # Inverse
    x_rec, log_det_inv = layer(y, cond, inverse=True)
    assert x_rec.shape == (batch, dim)
    assert log_det_inv.shape == (batch,)
    assert torch.isfinite(x_rec).all().item()
    assert torch.isfinite(log_det_inv).all().item()

    # Round-trip consistency
    assert torch.allclose(x_rec, x, rtol=1e-5, atol=1e-6)


def test_conditional_realnvp_log_prob_and_sample_shapes():
    torch.manual_seed(2)
    dim, cond_dim, batch, seq_len = 6, 4, 16, 3
    flow = ConditionalRealNVP(
        dim=dim, cond_dim=cond_dim, n_coupling_layers=4, hidden_dims=(32, 32)
    )

    u = torch.randn(batch, seq_len, dim)
    cond = torch.randn(batch, seq_len, cond_dim)

    # log_prob
    lp = flow.log_prob_seq(u, cond)
    assert lp.shape == (batch,)
    assert torch.isfinite(lp).all().item()

    # sample
    u_samp = flow.sample_seq(cond)
    assert u_samp.shape == (batch, seq_len, dim)
    assert torch.isfinite(u_samp).all().item()


def test_conditional_realnvp_round_trip_base_to_data():
    torch.manual_seed(3)
    dim, cond_dim, batch = 5, 3, 12
    flow = ConditionalRealNVP(
        dim=dim, cond_dim=cond_dim, n_coupling_layers=6, hidden_dims=(64, 64)
    )

    z = torch.randn(batch, dim)
    cond = torch.randn(batch, cond_dim)

    # Base -> data -> base
    u, log_det_f = flow.f(z, cond)
    z_rec, log_det_inv = flow.f_inv(u, cond)

    assert u.shape == (batch, dim)
    assert z_rec.shape == (batch, dim)
    assert log_det_f.shape == (batch,)
    assert log_det_inv.shape == (batch,)
    assert torch.isfinite(u).all().item()
    assert torch.isfinite(z_rec).all().item()
    assert torch.isfinite(log_det_f).all().item()
    assert torch.isfinite(log_det_inv).all().item()

    # Round-trip consistency
    assert torch.allclose(z_rec, z, rtol=1e-5, atol=1e-6)


def test_conditional_realnvp_round_trip_data_to_base():
    torch.manual_seed(4)
    dim, cond_dim, batch = 7, 5, 10
    flow = ConditionalRealNVP(
        dim=dim, cond_dim=cond_dim, n_coupling_layers=4, hidden_dims=(32, 32)
    )

    u = torch.randn(batch, dim)
    cond = torch.randn(batch, cond_dim)

    # Data -> base -> data
    z, log_det_inv = flow.f_inv(u, cond)
    u_rec, log_det_f = flow.f(z, cond)

    assert z.shape == (batch, dim)
    assert u_rec.shape == (batch, dim)
    assert log_det_inv.shape == (batch,)
    assert log_det_f.shape == (batch,)
    assert torch.isfinite(z).all().item()
    assert torch.isfinite(u_rec).all().item()
    assert torch.isfinite(log_det_inv).all().item()
    assert torch.isfinite(log_det_f).all().item()

    # Round-trip consistency
    assert torch.allclose(u_rec, u, rtol=1e-5, atol=1e-6)


def test_tail_transform_round_trip():
    torch.manual_seed(5)
    tail = TailTransformTTF(dim=3, init_lambda=0.2)
    z = torch.randn(8, 3)

    y, log_det = tail.forward(z)
    z_rec, log_det_inv = tail.inverse(y)

    assert y.shape == z.shape
    assert z_rec.shape == z.shape
    assert log_det.shape == (z.shape[0],)
    assert log_det_inv.shape == (z.shape[0],)
    assert torch.isfinite(y).all().item()
    assert torch.isfinite(z_rec).all().item()
    assert torch.allclose(z_rec, z, rtol=1e-5, atol=1e-6)
    assert torch.allclose(log_det + log_det_inv, torch.zeros_like(log_det), atol=1e-5)


def test_conditional_realnvp_with_flexible_tails_round_trip():
    torch.manual_seed(6)
    dim, cond_dim, batch = 4, 3, 6
    flow = ConditionalRealNVP(
        dim=dim,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(16, 16),
        use_flexible_tails=True,
        ttf_init_lambda=0.2,
    )

    z = torch.randn(batch, dim)
    cond = torch.randn(batch, cond_dim)

    u, _ = flow.f(z, cond)
    z_rec, _ = flow.f_inv(u, cond)

    assert torch.allclose(z_rec, z, rtol=1e-5, atol=1e-6)


def test_conditional_realnvp_config_includes_tail_flag():
    flow = ConditionalRealNVP(
        dim=3,
        cond_dim=2,
        n_coupling_layers=2,
        hidden_dims=(8, 8),
        use_flexible_tails=True,
    )
    cfg = flow.get_config()
    assert cfg["use_flexible_tails"] is True


def test_conditional_realnvp_with_arp_base_log_prob_seq():
    torch.manual_seed(7)
    dim, cond_dim, batch, seq_len = 3, 2, 5, 6
    flow = ConditionalRealNVP(
        dim=dim,
        cond_dim=cond_dim,
        n_coupling_layers=2,
        hidden_dims=(8, 8),
        base_dist=ARpBase(dim=dim, p=2, init_rho=[0.2, -0.05], init_sigma=0.7),
    )

    u = torch.randn(batch, seq_len, dim)
    cond = torch.randn(batch, seq_len, cond_dim)
    lp = flow.log_prob_seq(u, cond)

    assert lp.shape == (batch,)
    assert torch.isfinite(lp).all().item()
