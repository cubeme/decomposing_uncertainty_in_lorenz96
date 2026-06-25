import torch

from parameterization.flow.base_distribution import ARpBase, StdNormal


def test_std_normal_shapes_and_finite():
    base = StdNormal(dim=3)
    z = base.sample((4, 5, 3))
    lp = base.log_prob(z)

    assert z.shape == (4, 5, 3)
    assert lp.shape == (4, 5)
    assert torch.isfinite(z).all().item()
    assert torch.isfinite(lp).all().item()


def test_arp_base_log_prob_and_sample_shapes():
    base = ARpBase(dim=2, p=2, init_rho=[0.2, -0.05], init_sigma=0.3)
    z = base.sample((3, 8, 2))
    lp = base.log_prob(z)

    assert z.shape == (3, 8, 2)
    assert lp.shape == (3,)
    assert torch.isfinite(z).all().item()
    assert torch.isfinite(lp).all().item()


def test_arp_base_short_sequence_uses_init_only():
    base = ARpBase(dim=2, p=3, init_rho=[0.1, 0.0, -0.1], init_sigma=0.5)
    z = torch.randn(4, 2, 2)  # T < p
    lp = base.log_prob(z)

    assert lp.shape == (4,)
    assert torch.isfinite(lp).all().item()
