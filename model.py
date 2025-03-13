"""
Model definitions and builder functions using PyTorch Lightning.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any
from omegaconf import DictConfig
from utils import load_obj


def build_classifier(cfg: DictConfig, num_classes: int) -> nn.Module:
    """
    Build a classifier model based on the config.

    Args:
        cfg (DictConfig): Configuration object with model parameters.
        num_classes (int): Number of classes.

    Returns:
        nn.Module: The classifier model with the final layer adjusted.
    """
    backbone_cls = load_obj(cfg.model.backbone.class_name)
    model = backbone_cls(**cfg.model.backbone.params)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class LitClassifier(pl.LightningModule):
    """
    A PyTorch Lightning Module for classification.

    This module wraps a backbone model (built via build_classifier) and 
    implements training and validation steps along with optimizer configuration.
    """

    def __init__(self, cfg: DictConfig, model: nn.Module, num_classes: int) -> None:
        """
        Initialize the Lightning module.

        Args:
            cfg (DictConfig): The configuration object.
            model (nn.Module): The base classifier model.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch (Any): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch (Any): Batch of validation data.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss.
        """
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        logits = self(images)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)

        # Compute recall for the positive class (assuming label 1 is positive)
        true_positive = ((preds == 1) & (labels == 1)).sum().float()
        actual_positive = (labels == 1).sum().float()
        recall = true_positive / actual_positive if actual_positive > 0 else torch.tensor(0.0, device=self.device)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", (preds == labels).float().mean(), on_epoch=True, prog_bar=True)
        self.log("val_recall", recall, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.

        Returns:
            tuple: A tuple containing the optimizer and scheduler.
        """
        optimizer_cls = load_obj(self.cfg.optimizer.class_name)
        optimizer_params = self.cfg.optimizer.params.copy()
        optimizer_params.pop("gradient_clip_val", None)
        optimizer = optimizer_cls(self.model.parameters(), **optimizer_params)

        scheduler_cls = load_obj(self.cfg.scheduler.class_name)
        scheduler_params = self.cfg.scheduler.params
        scheduler = scheduler_cls(optimizer, **scheduler_params)

        return [optimizer], [{
            "scheduler": scheduler,
            "interval": self.cfg.scheduler.step,
            "monitor": self.cfg.scheduler.monitor
        }]
