#!/usr/bin/env python
"""Configuration settings and global variables."""

import os
from omegaconf import OmegaConf, DictConfig

CFG_DICT = {
    "run_mode": "train",  # "train" or "test"
    "tuning_mode": True,  # If True, tuning mode is active (pretrained_ckpt is ignored)
    "use_cv": True,       # If True, use cross-validation
    "use_optuna": True,   # If True, Optuna hyperparameter tuning is activated
    "general": {
        "save_dir": "baclogs4-3",
        "project_name": "bacteria"
    },
    "trainer": {
        "devices": 1,
        "accelerator": "auto",
        "precision": "16-mixed",
        "gradient_clip_val": 0.5
    },
    "training": {
        "seed": 666,
        "mode": "max",
        "tuning_epochs_detection": 2,
        "additional_epochs_detection": 2,
        "cross_validation": True,
        "num_folds": 2,
        "repeated_cv": 3,
        "composite_metric": {"alpha": 1.0, "beta": 0.1}
    },
    "optimizer": {
        "class_name": "torch.optim.AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.001,
            "gradient_clip_val": 0.0
        }
    },
    "scheduler": {
        "class_name": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "step": "epoch",
        "monitor": "val_loss",
        "params": {
            "mode": "min",
            "factor": 0.1,
            "patience": 10
        }
    },
    "model": {
        "backbone": {
            "class_name": "torchvision.models.resnet50",
            "params": {
                "weights": "IMAGENET1K_V1"
            }
        }
    },
    "data": {
        "detection_csv": "/kaggle/working/train.csv",
        "folder_path": "/kaggle/input/bacdataset/Data",
        "num_workers": 3,
        "batch_size": 4,
        "label_col": "label",
        "valid_split": 0.2
    },
    "augmentation": {
        "train": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Rotate", "params": {"limit": 10, "p": 0.5}},
                {"class_name": "albumentations.ColorJitter", "params": {"brightness": 0.1, "contrast": 0.1, "p": 0.1}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        },
        "valid": {
            "augs": [
                {"class_name": "albumentations.Resize", "params": {"height": 400, "width": 400, "p": 1.0}},
                {"class_name": "albumentations.Normalize", "params": {}},
                {"class_name": "albumentations.pytorch.transforms.ToTensorV2", "params": {"p": 1.0}}
            ]
        }
    },
    "test": {
        "folder_path": "/kaggle/input/bacdataset/Data/Test"
    },
    "test_csv": "None",
    "pretrained_ckpt": "/kaggle/working/baclogs3/20250315/best_model/best_detection.ckpt",
    "optuna": {
        "n_trials": 2,
        "params": {
            "lr": {"min": 1e-5, "max": 1e-3, "type": "loguniform"},
            "batch_size": {"values": [4, 8], "type": "categorical"},
            "gradient_clip_val": {"min": 0.0, "max": 0.3, "type": "float"},
            "weight_decay": {"min": 0.0, "max": 0.01, "type": "float"},
            "rotation_limit": {"min": 5, "max": 15, "type": "int"},
            "color_jitter_strength": {"min": 0.1, "max": 0.3, "type": "float"}
        }
    }
}

cfg: DictConfig = OmegaConf.create(CFG_DICT)

# This variable is used as a global path to save logs/checkpoints.
BASE_SAVE_DIR = None
