import pytest
import torch

from parameterization.flow.training.data import build_condition, make_paired_loader


def test_build_condition_with_history_and_forcing():
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    F = torch.tensor([10.0, 20.0, 30.0, 40.0])
    delta_t = 1
    si = 0.5
    dt_full = 1.0
    cond = build_condition(x, delta_t=delta_t, si=si, dt_full=dt_full, F=F)

    assert cond.shape == (4, 2 * (delta_t + 1) + 1)
    expected = torch.tensor(
        [
            [1.0, 2.0, 1.0, 2.0, 10.0],
            [3.0, 4.0, 1.0, 2.0, 20.0],
            [5.0, 6.0, 1.0, 2.0, 30.0],
            [7.0, 8.0, 3.0, 4.0, 40.0],
        ]
    )
    torch.testing.assert_close(cond, expected)


def test_make_paired_loader_aligns_with_history_and_forcing():
    x = torch.arange(6, dtype=torch.float32)  # (N,)
    u = torch.arange(6, dtype=torch.float32) + 100.0
    F = torch.arange(6, dtype=torch.float32) + 10.0
    delta_t = 2
    si = 0.1
    dt_full = 0.2

    loader = make_paired_loader(
        x=x,
        u=u,
        batch_size=16,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        delta_t=delta_t,
        si=si,
        dt_full=dt_full,
        F=F,
        seq_len=2,
    )
    cond, u_used = next(iter(loader))

    # number of sequences = T - seq_len + 1 = 5
    assert cond.shape == (5, 2, (delta_t + 1) + 1)
    assert u_used.shape == (5, 2, 1)

    u_expected = u.unfold(0, size=2, step=1)  # (5, 2)
    torch.testing.assert_close(u_used.squeeze(-1), u_expected)

    F_expected = F.unfold(0, size=2, step=1)  # (5, 2)
    torch.testing.assert_close(cond[:, :, -1], F_expected)

    expected_first_step = torch.tensor([0.0, 0.0, 0.0, 10.0])
    torch.testing.assert_close(cond[0, 0], expected_first_step)

    expected_second_step = torch.tensor([1.0, 0.0, 0.0, 11.0])
    torch.testing.assert_close(cond[0, 1], expected_second_step)

    expected_fifth = torch.tensor([4.0, 2.0, 0.0, 14.0])
    torch.testing.assert_close(cond[4, 0], expected_fifth)


def test_build_condition_invalid_inputs():
    x = torch.randn(4, 2)
    with pytest.raises(ValueError, match="delta_t must be >= 0"):
        build_condition(x, delta_t=-1)

    F_bad = torch.randn(4, 1)
    with pytest.raises(ValueError, match="F must be 1D"):
        build_condition(x, delta_t=0, F=F_bad)

    F_short = torch.randn(3)
    with pytest.raises(ValueError, match="F must have length N"):
        build_condition(x, delta_t=0, F=F_short)
