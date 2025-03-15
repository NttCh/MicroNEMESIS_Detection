#!/usr/bin/env python
"""Custom PyTorch Lightning Callbacks."""

import os
import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, fbeta_score
import openpyxl
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from config import BASE_SAVE_DIR
from utils import thai_time
import numpy as np


class CleanTQDMProgressBar(TQDMProgressBar):
    """A TQDM progress bar that does not leave previous bars behind."""

    def init_train_tqdm(self):
        """Initialize the training progress bar."""
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar


class TrialFoldProgressCallback(pl.Callback):
    """
    Callback to print trial/fold progress in a multi-trial, multi-fold setting.

    Args:
        trial_number (int): Current trial number.
        total_trials (int): Total number of trials.
        fold_number (int): Current fold number.
        total_folds (int): Total number of folds.
    """

    def __init__(
        self,
        trial_number: Optional[int] = None,
        total_trials: Optional[int] = None,
        fold_number: Optional[int] = None,
        total_folds: Optional[int] = None
    ):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.fold_number = fold_number
        self.total_folds = total_folds

    def on_train_start(self, trainer, pl_module):
        """Called when training starts."""
        msgs = []
        if self.trial_number is not None and self.total_trials is not None:
            msgs.append(f"Trial {self.trial_number}/{self.total_trials}")
        if self.fold_number is not None and self.total_folds is not None:
            msgs.append(f"Fold {self.fold_number}/{self.total_folds}")
        if msgs:
            print(" | ".join(msgs))


class PlotMetricsCallback(pl.Callback):
    """
    Callback to plot training/validation metrics at the end of training.

    Args:
        save_dir (str): Directory to save the plot.
    """

    def __init__(self, save_dir: Optional[str] = None) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.epochs: List[int] = []
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_recalls: List[float] = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each validation epoch."""
        epoch = trainer.current_epoch + 1
        self.epochs.append(epoch)

        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        val_recall = trainer.callback_metrics.get("val_recall")

        self.train_losses.append(train_loss.item() if train_loss else float('nan'))
        self.val_losses.append(val_loss.item() if val_loss else float('nan'))
        self.val_recalls.append(val_recall.item() if val_recall else float('nan'))

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of training."""
        global BASE_SAVE_DIR
        if self.save_dir is None:
            self.save_dir = os.path.join(BASE_SAVE_DIR, "plots")
        os.makedirs(self.save_dir, exist_ok=True)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(self.epochs, self.train_losses, label="Train Loss", marker="o")
        axs[0].plot(self.epochs, self.val_losses, label="Validation Loss", marker="o")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Loss vs. Epoch")
        axs[0].legend()
        axs[0].grid(True)

        axs[1].plot(self.epochs, self.val_recalls, label="Validation Recall", marker="o")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Recall")
        axs[1].set_title("Recall vs. Epoch")
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, "metrics_plot.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[PlotMetricsCallback] Saved metrics plot to {save_path}")


class OverallProgressCallback(pl.Callback):
    """Callback to display overall progress (epoch out of total epochs)."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the start of training."""
        self.total_epochs = trainer.max_epochs

    def on_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the start of each epoch."""
        epoch = trainer.current_epoch + 1
        remaining = self.total_epochs - trainer.current_epoch
        print(f"[OverallProgress] Epoch {epoch}/{self.total_epochs} - Remaining: {remaining}")


class MasterValidationMetricsCallback(pl.Callback):
    """
    Callback to compute and log validation metrics (accuracy, precision, recall, F2)
    into an Excel file after each validation epoch.
    """

    def __init__(self, base_dir: str, fold_number=None):
        super().__init__()
        self.excel_path = os.path.join(base_dir, "all_eval_metrics.xlsx")
        self.fold_number = fold_number if fold_number is not None else 0
        self.rows = []

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of each validation epoch."""
        pl_module.eval()
        val_loader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        all_preds, all_labels, all_loss, count = [], [], 0.0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(pl_module.device)
                labels = labels.to(pl_module.device)
                logits = pl_module(images)
                loss = criterion(logits, labels)
                all_loss += loss.item()
                count += 1
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = all_loss / count if count > 0 else 0.0
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f2 = fbeta_score(all_labels, all_preds, beta=2, average='weighted', zero_division=0)

        epoch = trainer.current_epoch + 1
        row = {
            'fold': self.fold_number,
            'epoch': epoch,
            'val_loss': avg_loss,
            'val_acc': acc,
            'val_prec': prec,
            'val_recall': rec,
            'val_f2': f2
        }
        self.rows.append(row)
        pl_module.train()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Called at the end of training."""
        if os.path.exists(self.excel_path):
            old_df = pd.read_excel(self.excel_path)
            new_df = pd.DataFrame(self.rows)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = pd.DataFrame(self.rows)

        combined.to_excel(self.excel_path, index=False)
        print(f"[MasterValidationMetricsCallback] Logged evaluation metrics to {self.excel_path}")


class OptunaCompositeReportingCallback(pl.Callback):
    """
    Callback to report a composite metric (val_recall - val_loss * beta) to Optuna and handle pruning.
    """

    def __init__(self, trial, cfg):
        super().__init__()
        self.trial = trial
        self.cfg = cfg

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of each validation epoch."""
        val_recall = trainer.callback_metrics.get("val_recall")
        val_loss = trainer.callback_metrics.get("val_loss")

        if val_recall is not None and val_loss is not None:
            alpha = self.cfg.training.composite_metric.alpha
            beta_ = self.cfg.training.composite_metric.beta
            composite = alpha * val_recall.item() - beta_ * val_loss.item()
            self.trial.report(composite, step=trainer.current_epoch)

            # If the trial should be pruned, raise an exception.
            from optuna.exceptions import TrialPruned
            if self.trial.should_prune():
                raise TrialPruned()
