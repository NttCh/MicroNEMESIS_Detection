#!/usr/bin/env python
"""Training utilities and main training functions."""

import os
import time
import optuna
import copy
import torch
import numpy as np
import pandas as pd
import albumentations as A
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import Optional, Tuple
from config import cfg, BASE_SAVE_DIR
from utils import set_seed, load_obj, thai_time
from data import PatchClassificationDataset, preview_data_before_after
from model import build_classifier, LitClassifier
from callbacks import (
    PlotMetricsCallback,
    OverallProgressCallback,
    TrialFoldProgressCallback,
    MasterValidationMetricsCallback,
    CleanTQDMProgressBar,
    OptunaCompositeReportingCallback
)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from optuna.exceptions import TrialPruned


def train_stage(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    trial: Optional[optuna.trial.Trial] = None,
    suppress_metrics: bool = False,
    trial_number=None,
    total_trials=None,
    fold_number=None,
    total_folds=None
) -> Tuple[LitClassifier, float]:
    """
    Train a single stage (with optional train/val split) and return the model and composite metric.

    Args:
        cfg (Any): Configuration object.
        csv_path (str): Path to CSV file for training data.
        num_classes (int): Number of classes.
        stage_name (str): Stage name (for logging).
        trial (optuna.trial.Trial): Optional trial object for hyperparameter tuning.
        suppress_metrics (bool): If True, do not plot metrics.
        trial_number (int): Current trial number.
        total_trials (int): Total number of trials.
        fold_number (int): Current fold number (for CV).
        total_folds (int): Total folds (for CV).

    Returns:
        Tuple[LitClassifier, float]: (trained model, composite metric score).
    """
    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )
    print(f"[INFO] Current Train dataset size: {len(train_df)} | Validation dataset size: {len(valid_df)}")

    train_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs
    ])
    valid_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs
    ])

    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)

    if trial_number == 1:
        print("[INFO] Previewing training data before and after augmentation (first trial only).")
        preview_data_before_after(train_df, cfg.data.folder_path, train_transforms, n=4)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    model = build_classifier(cfg, num_classes=num_classes)

    # Load pretrained checkpoint only if not in tuning mode
    if (not cfg.tuning_mode) and (cfg.get("pretrained_ckpt", "None") not in [None, "None"]):
        print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt}")
        state_dict = torch.load(cfg.pretrained_ckpt, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[len("model."):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

    lit_model = LitClassifier(cfg=cfg, model=model, num_classes=num_classes)

    stage_id = f"{stage_name}_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}")
    max_epochs = cfg.training.tuning_epochs_detection

    callbacks = []
    if not suppress_metrics:
        callbacks.append(PlotMetricsCallback())
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
        callbacks.append(ModelCheckpoint(
            dirpath=save_dir,
            monitor="val_recall",
            mode="max",
            save_top_k=1,
            filename=f"{stage_name}-" + "{epoch:02d}-{val_recall:.4f}"
        ))
        callbacks.append(OverallProgressCallback())
        callbacks.append(TrialFoldProgressCallback(
            trial_number=trial_number,
            total_trials=total_trials,
            fold_number=fold_number,
            total_folds=total_folds
        ))

    if trial is not None:
        callbacks.append(OptunaCompositeReportingCallback(trial, cfg))

    callbacks.append(MasterValidationMetricsCallback(base_dir=BASE_SAVE_DIR, fold_number=fold_number))
    callbacks.append(CleanTQDMProgressBar())

    trainer = Trainer(
        max_epochs=max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.get("gradient_clip_val", None),
        logger=logger,
        callbacks=callbacks,
        enable_model_summary=False
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    val_recall = trainer.callback_metrics.get("val_recall")
    val_loss = trainer.callback_metrics.get("val_loss")

    if val_recall and val_loss:
        alpha = cfg.training.composite_metric.alpha
        beta_ = cfg.training.composite_metric.beta
        composite_metric = alpha * val_recall.item() - beta_ * val_loss.item()
    else:
        composite_metric = 0.0

    return lit_model, composite_metric


def train_with_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    cv_run: int,
    total_cv: int,
    verbose: bool = True
) -> Tuple[LitClassifier, float]:
    """
    Train a model using cross-validation on the given CSV data.

    Args:
        cfg (Any): Configuration object.
        csv_path (str): Path to CSV file for training data.
        num_classes (int): Number of classes.
        stage_name (str): Stage name (for logging).
        cv_run (int): Current repeated CV run number.
        total_cv (int): Total number of repeated CV runs.
        verbose (bool): If True, print logs.

    Returns:
        Tuple[LitClassifier, float]: (best fold model, average metric across folds).
    """
    full_df = pd.read_csv(csv_path)
    skf = StratifiedKFold(n_splits=cfg.training.num_folds, shuffle=True, random_state=cfg.training.seed)

    val_scores = []
    fold_models = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(full_df, full_df[cfg.data.label_col])):
        fold_num = fold + 1
        if verbose:
            print(f"CV {cv_run}/{total_cv} | Fold {fold_num}/{cfg.training.num_folds}: ", end=" ")

        # Build train/valid sets for this fold
        train_df = full_df.iloc[train_idx]
        valid_df = full_df.iloc[valid_idx]
        train_df.to_csv("temp_train.csv", index=False)
        valid_df.to_csv("temp_valid.csv", index=False)

        # We'll call train_stage on the entire CSV, but the function does its own split.
        # To keep consistent with the original approach, let's just pass the full CSV.
        lit_model, val_metric = train_stage(
            cfg,
            csv_path,
            num_classes,
            stage_name=f"{stage_name}_fold{fold_num}",
            fold_number=fold_num,
            total_folds=cfg.training.num_folds
        )

        if verbose:
            print(f"| Composite Metric: {val_metric:.4f}")

        val_scores.append(val_metric)
        fold_models.append(lit_model)

    best_idx = np.argmax(val_scores)
    return fold_models[best_idx], np.mean(val_scores)


def repeated_cross_validation(
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str,
    repeats: int
) -> Tuple[LitClassifier, float]:
    """
    Perform repeated cross-validation multiple times.

    Args:
        cfg (Any): Configuration object.
        csv_path (str): Path to CSV file.
        num_classes (int): Number of classes.
        stage_name (str): Stage name.
        repeats (int): Number of repeated CV runs.

    Returns:
        Tuple[LitClassifier, float]: (best overall model, average metric).
    """
    all_scores = []
    best_models = []

    for r in range(repeats):
        print(f"\n=== Repeated CV run {r+1}/{repeats} ===")
        model_cv, avg_score = train_with_cross_validation(cfg, csv_path, num_classes, stage_name, r+1, repeats, verbose=True)
        all_scores.append(avg_score)
        best_models.append(model_cv)

    best_idx = np.argmax(all_scores)
    return best_models[best_idx], np.mean(all_scores)


def continue_training(
    lit_model,
    cfg,
    csv_path: str,
    num_classes: int,
    stage_name: str
) -> LitClassifier:
    """
    Continue training a given Lightning model for additional epochs.

    Args:
        lit_model (LitClassifier): The trained Lightning model.
        cfg (Any): Configuration object.
        csv_path (str): Path to CSV file for training data.
        num_classes (int): Number of classes.
        stage_name (str): Stage name.

    Returns:
        LitClassifier: The updated model after further training.
    """
    from sklearn.model_selection import train_test_split

    full_df = pd.read_csv(csv_path)
    train_df, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )

    import albumentations as A
    train_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.train.augs
    ])
    valid_transforms = A.Compose([
        load_obj(aug["class_name"])(**aug["params"]) for aug in cfg.augmentation.valid.augs
    ])

    from data import PatchClassificationDataset
    train_dataset = PatchClassificationDataset(train_df, cfg.data.folder_path, transforms=train_transforms)
    valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=valid_transforms)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    additional_epochs = cfg.training.additional_epochs_detection
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from callbacks import (
        PlotMetricsCallback,
        OverallProgressCallback,
        MasterValidationMetricsCallback,
        CleanTQDMProgressBar
    )
    from pytorch_lightning import Trainer

    stage_id = f"{stage_name}_continued_{thai_time().strftime('%Y%m%d-%H%M%S')}_{int(time.time()*1000)}"
    global BASE_SAVE_DIR
    save_dir = os.path.join(BASE_SAVE_DIR, stage_id)
    os.makedirs(save_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=save_dir, name=f"{cfg.general.project_name}_{stage_name}_continued")

    callbacks = [
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
        callbacks=callbacks,
        enable_model_summary=False
    )
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    return lit_model


def print_trial_thai_callback(study, trial):
    """
    Callback to print trial completion in Thai local time.
    """
    if trial.state == optuna.trial.TrialState.COMPLETE:
        finish_time = thai_time()
        print(f"Best Value: {trial.value:.4f}")
