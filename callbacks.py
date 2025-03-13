"""
Callback definitions for PyTorch Lightning training.
"""

import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             fbeta_score)
import seaborn as sns
import openpyxl
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage


class CleanTQDMProgressBar(TQDMProgressBar):
    """
    A custom TQDM progress bar that does not leave the bar after each epoch.
    """
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.leave = False
        return bar


class TrialFoldProgressCallback(Callback):
    """
    A callback to display trial/fold progress at the start of training.
    """
    def __init__(self, trial_number=None, total_trials=None,
                 fold_number=None, total_folds=None):
        super().__init__()
        self.trial_number = trial_number
        self.total_trials = total_trials
        self.fold_number = fold_number
        self.total_folds = total_folds

    def on_train_start(self, trainer, pl_module):
        msgs = []
        if self.trial_number is not None and self.total_trials is not None:
            msgs.append(f"Trial {self.trial_number}/{self.total_trials}")
        if self.fold_number is not None and self.total_folds is not None:
            msgs.append(f"Fold {self.fold_number}/{self.total_folds}")
        if msgs:
            print(" | ".join(msgs))


class PlotMetricsCallback(Callback):
    """
    A callback to plot training and validation metrics at the end of training.
    """
    def __init__(self):
        super().__init__()
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_recalls = []

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        self.epochs.append(epoch)
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        train_acc = trainer.callback_metrics.get("train_acc")
        val_acc = trainer.callback_metrics.get("val_acc")
        val_recall = trainer.callback_metrics.get("val_recall")

        if train_loss is not None:
            self.train_losses.append(train_loss.item())
        else:
            self.train_losses.append(float('nan'))

        if val_loss is not None:
            self.val_losses.append(val_loss.item())
        else:
            self.val_losses.append(float('nan'))

        if train_acc is not None:
            self.train_accs.append(train_acc.item())
        else:
            self.train_accs.append(float('nan'))

        if val_acc is not None:
            self.val_accs.append(val_acc.item())
        else:
            self.val_accs.append(float('nan'))

        if val_recall is not None:
            self.val_recalls.append(val_recall.item())
        else:
            self.val_recalls.append(float('nan'))

    def on_train_end(self, trainer, pl_module):
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
        save_path = os.path.join(trainer.logger.log_dir, "metrics_plot.png")
        plt.savefig(save_path)
        print(f"[PlotMetricsCallback] Saved metrics plot to {save_path}")
        plt.show()


class OverallProgressCallback(Callback):
    """
    A callback to display overall epoch progress at the start of each epoch.
    """
    def on_train_start(self, trainer, pl_module):
        self.total_epochs = trainer.max_epochs

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        remaining = self.total_epochs - trainer.current_epoch
        print(f"[OverallProgressCallback] Epoch {epoch}/{self.total_epochs} "
              f"- Remaining epochs: {remaining}")


class MasterValidationMetricsCallback(Callback):
    """
    A callback to record validation metrics (loss, acc, prec, recall, f2) in an Excel file.
    """
    def __init__(self, base_dir: str, fold_number=None):
        super().__init__()
        self.excel_path = os.path.join(base_dir, "all_eval_metrics.xlsx")
        self.fold_number = fold_number if fold_number is not None else 0
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        val_loader = trainer.val_dataloaders[0]
        all_preds = []
        all_labels = []
        all_loss = 0.0
        count = 0
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

    def on_train_end(self, trainer, pl_module):
        if os.path.exists(self.excel_path):
            old_df = pd.read_excel(self.excel_path)
            new_df = pd.DataFrame(self.rows)
            combined = pd.concat([old_df, new_df], ignore_index=True)
        else:
            combined = pd.DataFrame(self.rows)

        combined.to_excel(self.excel_path, index=False)
        print(f"[MasterValidationMetricsCallback] Logged validation metrics to {self.excel_path}")
