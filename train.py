"""
Training functions and utilities for cross-validation, repeated CV, etc.
"""

import copy
import time
import numpy as np
import torch
from typing import Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import DictConfig
import pandas as pd
import os

from data import PatchClassificationDataset, preview_data_before_after
from model import build_classifier, LitClassifier
from callbacks import (
    PlotMetricsCallback,
    OverallProgressCallback,
    TrialFoldProgressCallback,
    MasterValidationMetricsCallback,
    CleanTQDMProgressBar
)
import albumentations as A
from utils import load_obj, thai_time


def train_stage(
    cfg: DictConfig,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    trial: Optional["optuna.trial.Trial"] = None,
    suppress_metrics: bool = False,
    trial_number=None,
    total_trials=None,
    fold_number=None,
    total_folds=None
) -> Tuple["LitClassifier", float]:
    """
    Train a model for one stage (e.g. detection).
    Returns (model, composite_metric).
    """
    from sklearn.model_selection import train_test_split
    from models import LitClassifier

    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )

    # Build augmentation transforms
    train_transforms = A.Compose([
        getattr(__import__(aug["class_name"].rsplit('.', 1)[0],
                           fromlist=[aug["class_name"].rsplit('.', 1)[-1]]),
                aug["class_name"].rsplit('.', 1)[-1])(**aug["params"])
        for aug in cfg.augmentation.train.augs
    ])
    valid_transforms = A.Compose([
        getattr(__import__(aug["class_name"].rsplit('.', 1)[0],
                           fromlist=[aug["class_name"].rsplit('.', 1)[-1]]),
                aug["class_name"].rsplit('.', 1)[-1])(**aug["params"])
        for aug in cfg.augmentation.valid.augs
    ])

    train_dataset = PatchClassificationDataset(
        train_df,
        cfg.data.folder_path,
        transforms=train_transforms
    )
    valid_dataset = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=valid_transforms
    )

    print(f"[INFO] Train dataset size: {len(train_dataset)} | "
          f"Validation dataset size: {len(valid_dataset)}")

    if trial_number == 1 and not suppress_metrics:
        print("[INFO] Previewing training data before and after augmentation (first trial only).")
        preview_data_before_after(train_df, cfg.data.folder_path, train_transforms, n=4)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    # Build classifier
    from models import build_classifier
    model = build_classifier(cfg, num_classes=num_classes)

    # If there's a pretrained checkpoint
    if cfg.get("pretrained_ckpt", "None") != "None":
        print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt}")
        model.load_state_dict(torch.load(cfg.pretrained_ckpt))

    # Wrap in a LightningModule
    from pytorch_lightning import LightningModule
    import pytorch_lightning as pl

    class LitClassifier(pl.LightningModule):
        """
        A PyTorch Lightning model for classification.
        """
        def __init__(self, cfg, model, num_classes):
            super().__init__()
            self.cfg = cfg
            self.model = model
            self.criterion = torch.nn.CrossEntropyLoss()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

        def training_step(self, batch, batch_idx):
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

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self(images)
            loss = self.criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            true_positive = ((preds == 1) & (labels == 1)).sum().float()
            actual_positive = (labels == 1).sum().float()
            recall = (true_positive / actual_positive) if actual_positive > 0 \
                else torch.tensor(0.0, device=loss.device)

            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            self.log("val_acc", (preds == labels).float().mean(),
                     on_epoch=True, prog_bar=True)
            self.log("val_recall", recall, on_epoch=True, prog_bar=True)
            return loss

        def configure_optimizers(self):
            from utils import load_obj
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

    lit_model = LitClassifier(cfg, model, num_classes)

    stage_id = f"{stage_name}_{int(time.time()*1000)}"
    from pytorch_lightning.loggers import TensorBoardLogger
    from callbacks import (
        PlotMetricsCallback,
        OverallProgressCallback,
        TrialFoldProgressCallback,
        MasterValidationMetricsCallback,
        CleanTQDMProgressBar
    )

    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)
    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}")
    max_epochs = cfg.training.tuning_epochs_detection

    callbacks_list = []
    if not suppress_metrics:
        callbacks_list.append(PlotMetricsCallback())

    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    callbacks_list.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
    callbacks_list.append(ModelCheckpoint(
        dirpath=save_dir, monitor="val_recall", mode="max",
        save_top_k=1, filename=f"{stage_name}-" + "{epoch:02d}-{val_recall:.4f}"
    ))
    callbacks_list.append(OverallProgressCallback())
    callbacks_list.append(TrialFoldProgressCallback(
        trial_number=trial_number,
        total_trials=total_trials,
        fold_number=fold_number,
        total_folds=total_folds
    ))

    if trial is not None:
        class OptunaCompositeReportingCallback(pl.Callback):
            def __init__(self, trial):
                super().__init__()
                self.trial = trial

            def on_validation_epoch_end(self, trainer, pl_module):
                val_recall = trainer.callback_metrics.get("val_recall")
                val_loss = trainer.callback_metrics.get("val_loss")
                if val_recall is not None and val_loss is not None:
                    composite = val_recall.item() - val_loss.item()
                    self.trial.report(composite, step=trainer.current_epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()

        callbacks_list.append(OptunaCompositeReportingCallback(trial))

    callbacks_list.append(MasterValidationMetricsCallback(
        base_dir=BASE_SAVE_DIR,
        fold_number=fold_number
    ))
    callbacks_list.append(CleanTQDMProgressBar())

    from pytorch_lightning import Trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
        logger=logger,
        callbacks=callbacks_list,
        enable_model_summary=False
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    val_recall = trainer.callback_metrics.get("val_recall")
    val_loss = trainer.callback_metrics.get("val_loss")
    if val_recall is not None and val_loss is not None:
        composite_metric = val_recall.item() - val_loss.item()
    else:
        composite_metric = 0.0

    return lit_model, composite_metric


def train_with_cross_validation(
    cfg: DictConfig,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    verbose: bool = True
) -> Tuple["LitClassifier", float]:
    """
    Train with cross-validation and return the best model from the best fold
    along with the average score.
    """
    from sklearn.model_selection import StratifiedKFold
    from train import train_stage

    full_df = pd.read_csv(csv_path)
    skf = StratifiedKFold(
        n_splits=cfg.training.num_folds,
        shuffle=True,
        random_state=cfg.training.seed
    )

    val_scores = []
    fold_models = []
    if verbose:
        pbar = tqdm(total=cfg.training.num_folds, desc="CV Folds", leave=True)

    fold_idx = 0
    for train_idx, valid_idx in skf.split(full_df, full_df[cfg.data.label_col]):
        fold_idx += 1
        if verbose:
            print(f"Fold {fold_idx}/{cfg.training.num_folds}")

        train_df = full_df.iloc[train_idx]
        valid_df = full_df.iloc[valid_idx]

        lit_model, val_metric = train_stage(
            cfg,
            csv_path,
            num_classes,
            stage_name=f"{stage_name}_fold{fold_idx}",
            fold_number=fold_idx,
            total_folds=cfg.training.num_folds
        )
        score = val_metric
        if verbose:
            print(f"Fold {fold_idx} composite metric (recall - loss): {score:.4f}")

        val_scores.append(score)
        fold_models.append(lit_model)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()
        avg_score = float(np.mean(val_scores))
        print(f"Average CV composite metric: {avg_score:.4f}")
    else:
        avg_score = float(np.mean(val_scores))

    best_idx = int(np.argmax(val_scores))
    return fold_models[best_idx], avg_score


def repeated_cross_validation(
    cfg: DictConfig,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    repeats: int
) -> Tuple["LitClassifier", float]:
    """
    Perform repeated cross-validation and return the best model across runs
    and the overall average score.
    """
    all_scores = []
    best_models = []

    for r in range(repeats):
        print(f"\n=== Repeated CV run {r+1}/{repeats} ===")
        model_cv, avg_score = train_with_cross_validation(
            cfg, csv_path, num_classes, stage_name, verbose=True
        )
        all_scores.append(avg_score)
        best_models.append(model_cv)

    overall_avg = float(np.mean(all_scores))
    overall_std = float(np.std(all_scores))
    print(f"\nRepeated CV over {repeats} runs: {overall_avg:.4f} ± {overall_std:.4f}")

    best_idx = int(np.argmax(all_scores))
    return best_models[best_idx], overall_avg


def continue_training(
    lit_model: "LitClassifier",
    cfg: DictConfig,
    csv_path: str,
    num_classes: int,
    stage_name: str
) -> "LitClassifier":
    """
    Continue training an existing model for additional epochs.
    """
    from sklearn.model_selection import train_test_split
    from data import PatchClassificationDataset
    import albumentations as A
    import os

    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )

    train_transforms = A.Compose([
        getattr(__import__(aug["class_name"].rsplit('.', 1)[0],
                           fromlist=[aug["class_name"].rsplit('.', 1)[-1]]),
                aug["class_name"].rsplit('.', 1)[-1])(**aug["params"])
        for aug in cfg.augmentation.train.augs
    ])
    valid_transforms = A.Compose([
        getattr(__import__(aug["class_name"].rsplit('.', 1)[0],
                           fromlist=[aug["class_name"].rsplit('.', 1)[-1]]),
                aug["class_name"].rsplit('.', 1)[-1])(**aug["params"])
        for aug in cfg.augmentation.valid.augs
    ])

    train_dataset = PatchClassificationDataset(
        train_df,
        cfg.data.folder_path,
        transforms=train_transforms
    )
    valid_dataset = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=valid_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    from callbacks import (
        PlotMetricsCallback,
        OverallProgressCallback,
        MasterValidationMetricsCallback,
        CleanTQDMProgressBar
    )
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    additional_epochs = cfg.training.additional_epochs_detection
    from utils import thai_time
    stage_id = f"{stage_name}_continued_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"

    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_continued")

    callbacks_list = [
        PlotMetricsCallback(),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_recall",
            mode="max",
            filename=f"{stage_name}_continued-" + "{epoch:02d}-{val_recall:.4f}"
        ),
        OverallProgressCallback(),
        MasterValidationMetricsCallback(base_dir=BASE_SAVE_DIR),
        CleanTQDMProgressBar()
    ]

    trainer = Trainer(
        max_epochs=additional_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks_list,
        enable_model_summary=False
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model


def objective_stage(trial: "optuna.trial.Trial", stage: str) -> float:
    """
    Objective function for Optuna hyperparameter tuning.
    """
    import optuna
    from config import get_config
    from train import train_with_cross_validation
    from tqdm import tqdm

    cfg = get_config()
    trial_cfg = copy.deepcopy(cfg)

    for param_name, param_info in trial_cfg.optuna.params.items():
        if param_name in ["rotation_limit", "color_jitter_strength"]:
            continue

        ptype = param_info["type"]
        if ptype == "loguniform":
            trial_cfg.optimizer.params[param_name] = trial.suggest_float(
                param_name, param_info["min"], param_info["max"], log=True
            )
        elif ptype == "categorical":
            trial_cfg.data.batch_size = trial.suggest_categorical(
                param_name, param_info["values"]
            )
        elif ptype == "float":
            trial_cfg.optimizer.params[param_name] = trial.suggest_float(
                param_name, param_info["min"], param_info["max"]
            )
        elif ptype == "int":
            trial_cfg.optimizer.params[param_name] = trial.suggest_int(
                param_name, param_info["min"], param_info["max"]
            )

    rotation_limit = trial.suggest_int(
        "rotation_limit",
        trial_cfg.optuna.params["rotation_limit"]["min"],
        trial_cfg.optuna.params["rotation_limit"]["max"]
    )
    color_jitter_strength = trial.suggest_float(
        "color_jitter_strength",
        trial_cfg.optuna.params["color_jitter_strength"]["min"],
        trial_cfg.optuna.params["color_jitter_strength"]["max"]
    )

    for aug in trial_cfg.augmentation.train.augs:
        if aug["class_name"] == "albumentations.Rotate":
            aug["params"]["limit"] = rotation_limit
        elif aug["class_name"] == "albumentations.ColorJitter":
            aug["params"]["brightness"] = color_jitter_strength
            aug["params"]["contrast"] = color_jitter_strength

    trial_cfg.trainer.max_epochs = trial_cfg.training.tuning_epochs_detection
    csv_path = trial_cfg.data.detection_csv
    num_classes = 2
    scores = []

    with tqdm(total=trial_cfg.training.repeated_cv, desc="CV Trials", leave=False) as pbar:
        for _ in range(trial_cfg.training.repeated_cv):
            _, score = train_with_cross_validation(
                trial_cfg, csv_path, num_classes, "detection", verbose=False
            )
            scores.append(score)
            pbar.update(1)

    return float(np.mean(scores))
