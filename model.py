#!/usr/bin/env python
"""Model definitions (including the LightningModule)."""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from utils import load_obj


def build_classifier(cfg: Any, num_classes: int) -> nn.Module:
    """
    Build a classification model given a config.

    Args:
        cfg (DictConfig): The configuration object.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The PyTorch model.
    """
    backbone_cls = load_obj(cfg.model.backbone.class_name)
    model = backbone_cls(**cfg.model.backbone.params)

    # Replace the final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class LitClassifier(pl.LightningModule):
    """
    LightningModule wrapping a classifier model with training/validation steps.

    Args:
        cfg (Any): Configuration object.
        model (nn.Module): The PyTorch model.
        num_classes (int): Number of classes.
    """

    def __init__(self, cfg: Any, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        Args:
            batch (tuple): (images, labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Training loss.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train_loss", loss, on_epoch=True, prog_bar=False)
        self.log("train_acc", acc, on_epoch=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Perform a single validation step.

        Args:
            batch (tuple): (images, labels).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Validation loss.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        true_positive = ((preds == 1) & (labels == 1)).sum().float()
        actual_positive = (labels == 1).sum().float()
        recall = true_positive / actual_positive if actual_positive > 0 else torch.tensor(0.0, device=loss.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", (preds == labels).float().mean(), on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and scheduler from the config.

        Returns:
            tuple: (list of optimizers, list of schedulers)
        """
        optimizer_cls = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = self.cfg.optimizer.params.copy()
        if "gradient_clip_val" in optimizer_params:
            del optimizer_params["gradient_clip_val"]

        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)

        scheduler_cls = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = self.cfg.scheduler.params
        scheduler = scheduler_cls(optimizer, **scheduler_params)

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor
        }]
