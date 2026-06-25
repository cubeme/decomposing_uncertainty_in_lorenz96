"""Provide PyTorch time-stepping and model utilities."""

from typing import Callable

import torch


def l96_rhs_torch(x: torch.Tensor, F: float) -> torch.Tensor:
    """
    Torch implementation of Lorenz-96 eq1 RHS that supports batch (B, K)
    Matches standard L96:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    """
    x_ip1 = torch.roll(x, shifts=-1, dims=-1)
    x_im1 = torch.roll(x, shifts=1, dims=-1)
    x_im2 = torch.roll(x, shifts=2, dims=-1)
    return (x_ip1 - x_im2) * x_im1 - x + F


# Batched time stepping that mirrors euler_forward / RK2 / RK4
# but operates on torch tensors (B, K)
def euler_forward_torch(
    fn: Callable[[torch.Tensor, float], torch.Tensor],
    dt: float,
    x: torch.Tensor,
    t: float,
):
    return x + dt * fn(x, t)


def RK2_torch(
    fn: Callable[[torch.Tensor, float], torch.Tensor],
    dt: float,
    x: torch.Tensor,
    t: float,
):
    x_dot1 = fn(x, t)
    x_dot2 = fn(x + 0.5 * dt * x_dot1, t + 0.5 * dt)
    return x + dt * x_dot2


def RK4_torch(
    fn: Callable[[torch.Tensor, float], torch.Tensor],
    dt: float,
    x: torch.Tensor,
    t: float,
):
    k1 = fn(x, t)
    k2 = fn(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = fn(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = fn(x + dt * k3, t + dt)
    return x + (dt / 6.0) * ((k1 + k4) + 2.0 * (k2 + k3))
