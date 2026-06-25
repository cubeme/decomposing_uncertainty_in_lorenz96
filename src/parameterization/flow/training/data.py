"""Prepare Lorenz '96 data for normalizing-flow training."""

from __future__ import annotations

from typing import Iterable, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

TensorLike = Union[torch.Tensor, Iterable]


def as_tensor(x: TensorLike, device: torch.device) -> torch.Tensor:
    """Convert input to a torch.Tensor on the given device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, device=device)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure x has shape (N, D). If given a single vector (D,), treat it as (1, D).
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def ensure_time_series_2d(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Ensure time-series tensor has shape (N, D).

    For time-series inputs we want:
      - (N,)   -> (N, 1)
      - (N, D) -> unchanged

    This is intentionally different from ensure_2d, which interprets (D,) as (1, D).
    """
    if x.dim() == 1:
        return x.unsqueeze(1)  # (N,) -> (N, 1)
    if x.dim() == 2:
        return x
    raise ValueError(f"{name} must have shape (N,) or (N,D). Got {tuple(x.shape)}.")


def ensure_forcing_1d(F: torch.Tensor, N: int) -> torch.Tensor:
    """Validate forcing input F: (N,)."""
    if F.dim() != 1:
        raise ValueError(f"F must be 1D with shape (N,). Got {tuple(F.shape)}.")
    if F.shape[0] != N:
        raise ValueError(f"F must have length N={N}. Got {F.shape[0]}.")
    return F


def full_step_stride(si: float, dt_full: float = 1.0, *, tol: float = 1e-6) -> int:
    """
    Number of samples per "full" model step dt_full.
    Example: si=0.5, dt_full=1.0 -> stride=2.
    """
    if si <= 0:
        raise ValueError("si must be > 0.")
    if dt_full <= 0:
        raise ValueError("dt_full must be > 0.")

    stride_f = float(dt_full) / float(si)
    stride_i = int(round(stride_f))
    if abs(stride_f - stride_i) > tol:
        raise ValueError(f"dt_full/si must be (near) an integer. Got {stride_f}.")
    return stride_i


def build_condition(
    x: torch.Tensor,
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build conditioning features from a time series.

    For each time t, the condition includes x at time t and at the previous
    delta_t full time steps, and optionally the forcing at time t. Only exact
    full steps are used; intermediate samples are skipped. Missing past values
    at the beginning are padded by repeating the oldest available value.

    Args:
        x: Raw time-series input x(t), shape (N,) or (N, x_dim).
        delta_t: Number of full time steps to include in the conditioning history.
        si: Sampling interval of the data.
        dt_full: Physical length of one full time step.
        F: Optional scalar forcing time-series aligned with x.

    Returns:
        Conditioning tensor of shape (N, cond_dim).
    """
    if delta_t < 0:
        raise ValueError("delta_t must be >= 0.")

    stride = full_step_stride(si=si, dt_full=dt_full)
    x = ensure_time_series_2d(x, name="x")  # (N, x_dim)
    N, _ = x.shape

    parts = []
    for j in range(delta_t + 1):
        lag = j * stride  # <-- skip sub-steps
        if lag == 0:
            x_lag = x
        else:
            x_lag = torch.cat([x[:1].expand(lag, -1), x[:-lag]], dim=0)
        parts.append(x_lag)

    cond = torch.cat(parts, dim=-1)

    if F is not None:
        F = ensure_forcing_1d(F, N)
        cond = torch.cat([cond, F.unsqueeze(1)], dim=-1)

    return cond


class WindowDataset(Dataset):
    """
    Dataset of contiguous time windows.

    Each item is a pair (cond_window, u_window) with shapes:
      cond_window: (T, cond_dim)
      u_window:    (T, u_dim)
    """

    def __init__(self, cond: torch.Tensor, u: torch.Tensor, seq_len: int):
        super().__init__()
        if cond.dim() != 2 or u.dim() != 2:
            raise ValueError(
                "cond and u must be 2D time-series tensors:\n"
                "  cond: (N, cond_dim)\n"
                "  u:    (N, u_dim)"
            )
        if cond.shape[0] != u.shape[0]:
            raise ValueError("cond and u must have the same length N.")
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0.")

        self.cond = cond
        self.u = u
        self.seq_len = int(seq_len)
        self.N = int(cond.shape[0])

        if self.N < self.seq_len:
            raise ValueError(
                f"Not enough samples for seq_len={self.seq_len}. Got N={self.N}."
            )

        self.n_windows = self.N - self.seq_len + 1

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int):
        i0 = int(idx)
        i1 = i0 + self.seq_len
        return self.cond[i0:i1], self.u[i0:i1]


def make_paired_loader(
    x: TensorLike,
    u: TensorLike,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    num_workers: int = 0,
    delta_t: int = 0,
    si: float = 1.0,
    dt_full: float = 1.0,
    F: Optional[TensorLike] = None,
    seq_len: int = 32,
) -> DataLoader:
    """
    Create a DataLoader yielding paired (condition, target) sequence chunks.

    The condition is constructed from the raw time series. For each time t,
    it includes x at time t and at the previous delta_t full time steps, and
    optionally the forcing at time t. Missing past values are padded by
    repeating the oldest available value.

    The loader yields contiguous windows of length seq_len:
      cond: (B, T, cond_dim)
      u:    (B, T, u_dim)

    Args:
        x: Raw time-series input x(t), shape (N,) or (N, x_dim).
        u: Target time-series u(t), shape (N,) or (N, u_dim).
        batch_size: Batch size (number of windows per batch).
        shuffle: Whether to shuffle windows.
        pin_memory: Whether to pin memory for GPU transfer.
        num_workers: Number of DataLoader workers.
        delta_t: Number of full time steps in the conditioning history.
        si: Sampling interval of the data.
        dt_full: Physical length of one full time step.
        F: Optional scalar forcing time-series aligned with x.
        seq_len: Length of each contiguous time window.

    Returns:
        DataLoader yielding (condition, target) batches of shape (B, T, *).
    """
    x_tensor = as_tensor(x, torch.device("cpu"))
    u_tensor = as_tensor(u, torch.device("cpu"))

    x_tensor = ensure_time_series_2d(x_tensor, name="x")
    u_tensor = ensure_time_series_2d(u_tensor, name="u")

    if x_tensor.shape[0] != u_tensor.shape[0]:
        raise ValueError("x and u must have the same number of samples (same N).")

    F_tensor: Optional[torch.Tensor] = None
    if F is not None:
        F_tensor = as_tensor(F, torch.device("cpu"))
        F_tensor = ensure_forcing_1d(F_tensor, x_tensor.shape[0])

    cond = build_condition(
        x_tensor, delta_t=delta_t, si=si, dt_full=dt_full, F=F_tensor
    )

    dataset = WindowDataset(cond, u_tensor, seq_len=seq_len)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
