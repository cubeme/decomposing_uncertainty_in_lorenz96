import numpy as np
import torch

from ensemble.gcm_flow_param_batched import BatchedFlowParameterizationSharedFlow


class FakeFlow:
    """Minimal stand-in for ConditionalRealNVP."""

    def __init__(self, cond_dim: int, dim: int, offset: float):
        self.cond_dim = cond_dim
        self.dim = dim
        self.offset = float(offset)
        self.last_cond = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def sample_seq(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # store condition for inspection in tests
        self.last_cond = cond.detach().cpu()
        return z + self.offset


def test_models_rhos_seeds_used_correctly_across_N_M():
    device = torch.device("cpu")
    N, M, L, x_dim, u_dim = 3, 2, 1, 4, 4
    B = N * M * L
    x = torch.randn(B, x_dim, device=device)

    # Helper: make seeds with chosen broadcast pattern
    def seeds_all_same(val: int) -> np.ndarray:
        return np.full((N, M, L), val, dtype=np.int64)

    def seeds_per_member(s0: int, s1: int) -> np.ndarray:
        # broadcast across N and L, vary across member index
        s = np.empty((N, M, L), dtype=np.int64)
        s[:, 0, :] = s0
        s[:, 1, :] = s1
        return s

    # --- 1) Same seeds everywhere => same z across all realizations (deterministic here) ---
    flow = FakeFlow(x_dim, u_dim, offset=0.0)
    seeds = seeds_all_same(123)
    rho = 0.0

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=0,
        rho=rho,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_dim,
    )
    u = p.predict(x).view(N, M, L, u_dim)

    # within each member: broadcast across N (because we constructed seeds that way)
    assert torch.allclose(u[0, 0, 0], u[1, 0, 0]) and torch.allclose(
        u[1, 0, 0], u[2, 0, 0]
    )
    assert torch.allclose(u[0, 1, 0], u[1, 1, 0]) and torch.allclose(
        u[1, 1, 0], u[2, 1, 0]
    )

    # between members: same seed => same z => identical outputs
    assert torch.allclose(u[:, 0, 0, :], u[:, 1, 0, :])

    # --- 2) Seeds vary per member, broadcast across N (same flow => only seeds differ) ---
    flow = FakeFlow(x_dim, u_dim, offset=0.0)
    seeds = seeds_per_member(111, 222)
    rho = 0.0

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=0,
        rho=rho,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_dim,
    )
    u = p.predict(x).view(N, M, L, u_dim)

    # broadcast across N per member
    assert torch.allclose(u[0, 0, 0], u[1, 0, 0]) and torch.allclose(
        u[1, 0, 0], u[2, 0, 0]
    )
    assert torch.allclose(u[0, 1, 0], u[1, 1, 0]) and torch.allclose(
        u[1, 1, 0], u[2, 1, 0]
    )

    # different seeds => members differ (deterministic here with FakeFlow)
    assert not torch.allclose(u[:, 0, 0, :], u[:, 1, 0, :])

    # --- 3) rho behavior: AR(0) draws fresh z each call; AR(1) reuses z until update() ---
    flow = FakeFlow(x_dim, u_dim, offset=0.0)
    seeds = seeds_all_same(999)

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=0,
        rho=0.0,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_dim,
    )
    u1 = p.predict(x).view(N, M, L, u_dim)
    u2 = p.predict(x).view(N, M, L, u_dim)  # no update in between
    assert not torch.allclose(u1, u2)

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=1,
        rho=0.9,
        sigma=0.2,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_dim,
    )
    u1 = p.predict(x).view(N, M, L, u_dim)
    u2 = p.predict(x).view(N, M, L, u_dim)  # no update in between
    assert torch.allclose(u1, u2)

    p.update()
    u3 = p.predict(x).view(N, M, L, u_dim)
    assert not torch.allclose(u2, u3)


def test_batched_flow_condition_includes_history_and_forcing():
    device = torch.device("cpu")
    N, M, L, x_base_dim, u_dim = 1, 1, 1, 2, 3
    delta_t = 1
    si = 0.5
    dt_full = 1  # ensures stride=2 if full_step_stride(si, dt_full)=2
    cond_dim = x_base_dim * (delta_t + 1) + 1
    B = N * M * L

    flow = FakeFlow(cond_dim, u_dim, offset=0.0)
    seeds = np.full((N, M, L), 123, dtype=np.int64)

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=0,
        rho=0.0,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_base_dim,
        delta_t=delta_t,
        si=si,
        dt_full=dt_full,
        include_forcing_in_cond=True,
    )

    x_prev2 = torch.tensor([[1.0, 2.0]], device=device)
    x_prev1 = torch.tensor([[3.0, 4.0]], device=device)
    p._x_hist = [x_prev2, x_prev1]

    x_cur = torch.tensor([[5.0, 6.0]], device=device)
    p.predict(x_cur, F=30.0)

    assert flow.last_cond is not None
    # In current implementation cond_bt has shape (B, 1, cond_dim)
    assert flow.last_cond.shape == (B, 1, cond_dim)

    expected = torch.tensor([5.0, 6.0, 1.0, 2.0, 30.0])
    torch.testing.assert_close(flow.last_cond[0, 0], expected)


def test_batched_flow_ar_p_update_changes_state():
    device = torch.device("cpu")
    N, M, L, x_dim, u_dim = 2, 2, 1, 3, 3
    B = N * M * L
    x = torch.randn(B, x_dim, device=device)
    flow = FakeFlow(x_dim, u_dim, offset=0.0)

    # seeds broadcast across N, vary across member (like original test intent)
    seeds = np.empty((N, M, L), dtype=np.int64)
    seeds[:, 0, :] = 42
    seeds[:, 1, :] = 43

    p = BatchedFlowParameterizationSharedFlow(
        flow_model=flow,
        ar_order=2,
        rho=np.array([0.4, -0.1]),
        sigma=0.5,
        seeds=seeds,
        N=N,
        M=M,
        L=L,
        device=device,
        x_base_dim=x_dim,
    )

    u1 = p.predict(x)
    p.update()
    u2 = p.predict(x)
    assert not torch.allclose(u1, u2)
