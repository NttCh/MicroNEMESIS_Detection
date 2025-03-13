"""
Configuration dictionary and methods for loading config.
"""

import datetime
import random
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig

CONFIG_DICT = {
    "stage_detection": True,
    "general": {
        "save_dir": "baclogs3",
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
        "tuning_epochs_detection": 10,
        "additional_epochs_detection": 10,
        "cross_validation": True,
        "num_folds": 2,
        "repeated_cv": 3
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
                {
                    "class_name": "albumentations.Resize",
                    "params": {"height": 400, "width": 400, "p": 1.0}
                },
                {
                    "class_name": "albumentations.Rotate",
                    "params": {"limit": 10, "p": 0.5}
                },
                {
                    "class_name": "albumentations.ColorJitter",
                    "params": {"brightness": 0.1, "contrast": 0.1, "p": 0.1}
                },
                {
                    "class_name": "albumentations.Normalize",
                    "params": {}
                },
                {
                    "class_name": "albumentations.pytorch.transforms.ToTensorV2",
                    "params": {"p": 1.0}
                }
            ]
        },
        "valid": {
            "augs": [
                {
                    "class_name": "albumentations.Resize",
                    "params": {"height": 400, "width": 400, "p": 1.0}
                },
                {
                    "class_name": "albumentations.Normalize",
                    "params": {}
                },
                {
                    "class_name": "albumentations.pytorch.transforms.ToTensorV2",
                    "params": {"p": 1.0}
                }
            ]
        }
    },
    "test": {
        "folder_path": "None"
    },
    "optuna": {
        "use_optuna": True,
        "n_trials": 20,
        "params": {
            "lr": {"min": 1e-5, "max": 1e-3, "type": "loguniform"},
            "batch_size": {"values": [4, 8], "type": "categorical"},
            "gradient_clip_val": {"min": 0.0, "max": 0.3, "type": "float"},
            "weight_decay": {"min": 0.0, "max": 0.01, "type": "float"},
            "rotation_limit": {"min": 5, "max": 15, "type": "int"},
            "color_jitter_strength": {"min": 0.1, "max": 0.3, "type": "float"}
        }
    },
    "pretrained_ckpt": "None"
}


def get_config() -> DictConfig:
    """
    Return the default configuration as a DictConfig object.
    """
    return OmegaConf.create(CONFIG_DICT)
