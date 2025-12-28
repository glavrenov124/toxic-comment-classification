from __future__ import annotations

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score

from .models import TextCNN, TextCNNConfig


class ToxicLitModule(LightningModule):
    def __init__(
        self,
        model_cfg: TextCNNConfig,
        pad_id: int,
        lr: float,
        weight_decay: float,
        threshold: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pad_id"])
        self.model = TextCNN(model_cfg, pad_id=pad_id)
        self.loss_fn = nn.BCEWithLogitsLoss()

        num_labels = model_cfg.num_labels
        self.val_micro_f1 = MultilabelF1Score(
            num_labels=num_labels, average="micro", threshold=threshold
        )
        self.val_macro_f1 = MultilabelF1Score(
            num_labels=num_labels, average="macro", threshold=threshold
        )
        self.val_macro_auc = MultilabelAUROC(num_labels=num_labels, average="macro")

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    def training_step(self, batch, batch_idx: int):
        token_ids, labels = batch
        logits = self(token_ids)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int):
        token_ids, labels = batch
        logits = self(token_ids)
        loss = self.loss_fn(logits, labels)
        probs = torch.sigmoid(logits)

        self.val_micro_f1.update(probs, labels.int())
        self.val_macro_f1.update(probs, labels.int())
        self.val_macro_auc.update(probs, labels.int())

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/micro_f1", self.val_micro_f1.compute(), prog_bar=True)
        self.log("val/macro_f1", self.val_macro_f1.compute(), prog_bar=False)
        self.log("val/macro_auc", self.val_macro_auc.compute(), prog_bar=False)

        self.val_micro_f1.reset()
        self.val_macro_f1.reset()
        self.val_macro_auc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
