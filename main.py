#!/usr/bin/env python
"""
Main script to run training or testing based on the config settings.
"""

import os
import sys
import torch
import optuna
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from config import cfg, BASE_SAVE_DIR
from utils import set_seed, thai_time
from model import build_classifier
from train import (
    train_stage,
    repeated_cross_validation,
    continue_training,
    print_trial_thai_callback
)
from inference import (
    evaluate_model,
    display_sample_predictions,
    predict_test_folder,
    evaluate_test_roc,
    apply_compose
)
from data import PatchClassificationDataset
from optuna.exceptions import TrialPruned
from pytorch_lightning import seed_everything


def main():
    """
    Main entry point for running the script.
    Depending on cfg.run_mode, either trains or tests a model.
    """
    # Set random seed
    set_seed(cfg.training.seed)
    seed_everything(cfg.training.seed, workers=True)

    # Create the base directory for logs if it doesn't exist
    date_folder = thai_time().strftime("%Y%m%d")
    global BASE_SAVE_DIR
    BASE_SAVE_DIR = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Directory to store best model
    best_model_folder = os.path.join(BASE_SAVE_DIR, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)

    if cfg.run_mode.lower() == "test":
        print("[Main] TEST ONLY MODE")

        if not cfg.pretrained_ckpt:
            print("Please provide a valid pretrained checkpoint for testing.")
            sys.exit(1)

        num_classes = 2
        model = build_classifier(cfg, num_classes)

        print(f"Loading pretrained checkpoint from {cfg.pretrained_ckpt}")
        state_dict = torch.load(cfg.pretrained_ckpt, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[len("model."):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        valid_transform = A.Compose([
            A.Resize(height=400, width=400, p=1.0),
            A.Normalize(),
            A.pytorch.transforms.ToTensorV2(p=1.0)
        ])

        test_folder = cfg.test.folder_path
        output_excel = os.path.join(BASE_SAVE_DIR, "test_predictions.xlsx")
        predict_test_folder(model, test_folder, valid_transform, output_excel, print_results=True, model_path="None")

        if cfg.test_csv != "None":
            print("[Main] Evaluating ROC curve on test CSV")
            evaluate_test_roc(model, cfg.test_csv, cfg.test.folder_path, valid_transform)

        # Show a few sample predictions from test folder (optional)
        # ...

        print("[Main] TEST ONLY option complete.")

    else:
        print("[Main] TRAINING MODE")
        detection_csv = cfg.data.detection_csv

        # If in tuning mode + optuna
        if cfg.tuning_mode and cfg.use_optuna:
            def objective(trial):
                trial_cfg = cfg.copy()
                trial_cfg.tuning_mode = True  # ignore pretrained_ckpt
                optuna_params = trial_cfg.get("optuna", {}).get("params", {})

                # Adjust hyperparameters from the trial
                for param_name, param_info in optuna_params.items():
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

                rotation_min = optuna_params.get("rotation_limit", {}).get("min", 5)
                rotation_max = optuna_params.get("rotation_limit", {}).get("max", 15)
                color_min = optuna_params.get("color_jitter_strength", {}).get("min", 0.1)
                color_max = optuna_params.get("color_jitter_strength", {}).get("max", 0.3)

                rotation_limit = trial.suggest_int("rotation_limit", rotation_min, rotation_max)
                color_jitter_strength = trial.suggest_float("color_jitter_strength", color_min, color_max)

                # Update augmentation parameters
                for aug in trial_cfg.augmentation.train.augs:
                    if aug["class_name"] == "albumentations.Rotate":
                        aug["params"]["limit"] = rotation_limit
                    elif aug["class_name"] == "albumentations.ColorJitter":
                        aug["params"]["brightness"] = color_jitter_strength
                        aug["params"]["contrast"] = color_jitter_strength

                # Run training
                trial_cfg.trainer.max_epochs = trial_cfg.training.tuning_epochs_detection
                num_classes = 2

                if trial_cfg.get("use_cv", False):
                    _, score = repeated_cross_validation(
                        trial_cfg,
                        trial_cfg.data.detection_csv,
                        num_classes,
                        "detection",
                        repeats=trial_cfg.training.repeated_cv
                    )
                else:
                    _, score = train_stage(
                        trial_cfg,
                        trial_cfg.data.detection_csv,
                        num_classes,
                        "detection"
                    )
                return score

            n_trials = cfg.get("optuna", {}).get("n_trials", 1)
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[print_trial_thai_callback])

            print("[Optuna] Best trial value:", study.best_trial.value)

            # Save all trials to an Excel file
            eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
            os.makedirs(eval_folder, exist_ok=True)
            all_trials_df = study.trials_dataframe()
            trials_excel_path = os.path.join(eval_folder, "optuna_trials.xlsx")
            all_trials_df.to_excel(trials_excel_path, index=False)
            print(f"[Optuna] Saved all trials to {trials_excel_path}")

            # Save best trial params
            best_params = study.best_trial.params
            best_value = study.best_trial.value
            df_params = pd.DataFrame([{**best_params, "best_value": best_value}])
            best_params_path = os.path.join(eval_folder, "optuna_best_params.xlsx")
            df_params.to_excel(best_params_path, index=False)
            print(f"[Optuna] Saved best trial params to {best_params_path}")

            # (Optional) Save visualizations
            import optuna.visualization as vis
            vis.plot_optimization_history(study).write_image(os.path.join(eval_folder, "opt_history.png"))
            vis.plot_param_importances(study).write_image(os.path.join(eval_folder, "param_importance.png"))
            vis.plot_slice(study, params=list(best_params.keys())).write_image(os.path.join(eval_folder, "slice.png"))
            print(f"[Optuna] Saved study plots to {eval_folder}")

        # Now do the main training with or without CV
        if cfg.get("use_cv", False):
            detection_model, detection_metric = repeated_cross_validation(
                cfg,
                detection_csv,
                2,
                "detection",
                repeats=cfg.training.repeated_cv
            )
        else:
            detection_model, detection_metric = train_stage(
                cfg,
                detection_csv,
                2,
                "detection",
                trial_number=None
            )

        # Save detection checkpoint
        detection_checkpoint = os.path.join(best_model_folder, "best_detection.ckpt")
        torch.save(detection_model.state_dict(), detection_checkpoint)
        print(f"[Main] Saved detection checkpoint to {detection_checkpoint}")

        # Continue training if needed
        detection_model = continue_training(detection_model, cfg, detection_csv, 2, "detection")

        # Evaluate final model
        from inference import evaluate_model, display_sample_predictions
        evaluate_model(detection_model, detection_csv, cfg, stage="Detection")

        # Display sample predictions from validation set
        from sklearn.model_selection import train_test_split
        full_df = pd.read_csv(detection_csv)
        _, valid_df = train_test_split(
            full_df,
            test_size=cfg.data.valid_split,
            random_state=cfg.training.seed,
            stratify=full_df[cfg.data.label_col]
        )

        valid_augs = []
        for aug in cfg.augmentation.valid.augs:
            valid_augs.append(A.__dict__[aug["class_name"]](**aug["params"]))

        valid_dataset = PatchClassificationDataset(valid_df, cfg.data.folder_path, transforms=A.Compose(valid_augs))
        display_sample_predictions(detection_model, valid_dataset, num_samples=4)

        # Optionally predict on a test folder
        if str(cfg.test.folder_path).lower() != "none":
            test_folder = cfg.test.folder_path
            output_excel = os.path.join(BASE_SAVE_DIR, "test_predictions.xlsx")
            print("\n[Main] Predicting on test folder:")
            predict_test_folder(detection_model, test_folder, A.Compose(valid_augs), output_excel, print_results=True, model_path=cfg.pretrained_ckpt)

        print("[Main] Process finished.")


if __name__ == "__main__":
    main()
