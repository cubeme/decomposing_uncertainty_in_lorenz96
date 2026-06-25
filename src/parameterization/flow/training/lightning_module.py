"""Define the Lightning module for normalizing-flow training."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from parameterization.flow.flow_model import ConditionalRealNVP


class FlowLightningModule(pl.LightningModule):
    """
    Lightning wrapper around ConditionalRealNVP.

    Expects batches:
        cond : (B, T, cond_dim)
        u    : (B, T, dim)

    Loss:
        NLL = -E[ log p(u | cond) ]
    """

    def __init__(
        self, model: ConditionalRealNVP, lr: float, weight_decay: float
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        self.save_hyperparameters(ignore=["model"])

    def _nll(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if cond.dim() != 3 or u.dim() != 3:
            raise ValueError(
                "Expected sequence batches:\n"
                "  cond : (B, T, cond_dim)\n"
                "  u    : (B, T, dim)"
            )

        log_prob = self.model.log_prob_seq(u, cond)  # (B,)

        T = u.shape[1]
        D = u.shape[2]
        loss = -(log_prob / (T * D)).mean()
        return loss

    def training_step(self, batch, batch_idx):
        cond_batch, u_batch = batch
        loss = self._nll(cond_batch, u_batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        cond_batch, u_batch = batch
        loss = self._nll(cond_batch, u_batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def test_step(self, batch, batch_idx):
        cond_batch, u_batch = batch
        loss = self._nll(cond_batch, u_batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
