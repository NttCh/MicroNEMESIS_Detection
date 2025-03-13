"""
Main entry point for detection training and evaluation.
"""

import os
import torch
import pandas as pd
import albumentations as A
from omegaconf import DictConfig
from typing import Optional
from config import get_config
from utils import set_random_seed, thai_time
from data import PatchClassificationDataset
from train import (
    train_with_cross_validation,
    repeated_cross_validation,
    train_stage,
    continue_training
)
from inference import evaluate_model, display_sample_predictions, predict_test_folder


def main():
    """
    Main entry point for detection stage training and evaluation.
    """
    cfg = get_config()
    set_random_seed(cfg.training.seed)

    date_folder = thai_time().strftime("%Y%m%d")
    base_save_dir = os.path.join(cfg.general.save_dir, date_folder)
    os.makedirs(base_save_dir, exist_ok=True)
    best_model_folder = os.path.join(base_save_dir, "best_model")
    os.makedirs(best_model_folder, exist_ok=True)

    pretrained_ckpt = cfg.get("pretrained_ckpt", "None")

    if cfg.stage_detection:
        print("[Main] Training Stage 1: Detection (binary classification)")
        detection_csv = cfg.data.detection_csv

        # Branch for cross-validation + optuna
        if cfg.training.cross_validation and cfg.optuna.use_optuna:
            import optuna
            from tqdm import tqdm
            from train import objective_stage

            def objective(trial: optuna.trial.Trial) -> float:
                return objective_stage(trial, "detection")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
            print("[Optuna] Best trial:", study.best_trial)

            trials_df = study.trials_dataframe()
            eval_folder = os.path.join(base_save_dir, "eval")
            os.makedirs(eval_folder, exist_ok=True)
            optuna_excel_path = os.path.join(eval_folder, "optuna_trials.xlsx")
            trials_df.to_excel(optuna_excel_path, index=False)
            print(f"[Optuna] Results saved to {optuna_excel_path}")

            import optuna.visualization as vis
            vis.plot_optimization_history(study).write_image(
                os.path.join(eval_folder, "opt_history.png")
            )
            vis.plot_param_importances(study).write_image(
                os.path.join(eval_folder, "param_importance.png")
            )
            vis.plot_slice(study, params=list(study.best_trial.params.keys())).write_image(
                os.path.join(eval_folder, "slice.png")
            )

            # Update cfg with best trial parameters
            best_params = study.best_trial.params
            for k, v in best_params.items():
                if k == "batch_size":
                    cfg.data.batch_size = v
                elif k in ["lr", "weight_decay", "gradient_clip_val"]:
                    cfg.optimizer.params[k] = v
                elif k == "rotation_limit":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.Rotate":
                            aug["params"]["limit"] = v
                elif k == "color_jitter_strength":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.ColorJitter":
                            aug["params"]["brightness"] = v
                            aug["params"]["contrast"] = v

            from train import repeated_cross_validation
            detection_model, final_metric = repeated_cross_validation(
                cfg, detection_csv, 2, "detection", repeats=cfg.training.repeated_cv
            )
            print(f"[Main] Final repeated CV average composite metric (recall - loss): {final_metric:.4f}")

        # Branch for cross-validation (no optuna)
        elif cfg.training.cross_validation and not cfg.optuna.use_optuna:
            from train import train_with_cross_validation

            def repeated_cv(config, csv_file, num_classes, stage_name):
                scores = []
                for r in range(config.training.repeated_cv):
                    print(f"[Main] Repeated CV run {r+1}/{config.training.repeated_cv}")
                    _, score = train_with_cross_validation(
                        config, csv_file, num_classes, stage_name
                    )
                    scores.append(score)
                return float(sum(scores) / len(scores))

            avg_metric = repeated_cv(cfg, detection_csv, 2, "detection")
            print(f"[Main] Repeated CV average composite metric: {avg_metric:.4f}")
            from train import train_with_cross_validation
            detection_model, _ = train_with_cross_validation(cfg, detection_csv, 2, "detection")

        # Branch for single train + optuna
        elif not cfg.training.cross_validation and cfg.optuna.use_optuna:
            import optuna
            from tqdm import tqdm
            from train import objective_stage

            def objective(trial):
                return objective_stage(trial, "detection")

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cfg.optuna.n_trials, show_progress_bar=True)
            best_params = study.best_trial.params
            for k, v in best_params.items():
                if k == "batch_size":
                    cfg.data.batch_size = v
                elif k in ["lr", "weight_decay", "gradient_clip_val"]:
                    cfg.optimizer.params[k] = v
                elif k == "rotation_limit":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.Rotate":
                            aug["params"]["limit"] = v
                elif k == "color_jitter_strength":
                    for aug in cfg.augmentation.train.augs:
                        if aug["class_name"] == "albumentations.ColorJitter":
                            aug["params"]["brightness"] = v
                            aug["params"]["contrast"] = v

            from train import train_stage
            detection_model, detection_metric = train_stage(cfg, detection_csv, 2, "detection", trial_number=1)
            print(f"[Main] Final tuned composite metric (recall - loss): {detection_metric:.4f}")

        # Branch for single train, no CV, no optuna
        else:
            print("[Main] Doing single train (no CV, no optuna).")
            from train import train_stage
            detection_model, detection_metric = train_stage(cfg, detection_csv, 2, "detection", trial_number=None)

        # Save checkpoint
        detection_checkpoint = os.path.join(best_model_folder, "best_detection.ckpt")
        torch.save(detection_model.state_dict(), detection_checkpoint)
        print(f"[Main] Saved detection checkpoint to {detection_checkpoint}")

        # Continue training
        from train import continue_training
        detection_model = continue_training(detection_model, cfg, detection_csv, 2, "detection")

        # Evaluate
        from inference import evaluate_model
        evaluate_model(detection_model, detection_csv, cfg, stage="Detection")

        # Show sample predictions
        from sklearn.model_selection import train_test_split
        full_df = pd.read_csv(detection_csv)
        _, valid_df = train_test_split(
            full_df,
            test_size=cfg.data.valid_split,
            random_state=cfg.training.seed,
            stratify=full_df[cfg.data.label_col]
        )

        valid_transforms = A.Compose([
            getattr(__import__(aug["class_name"].rsplit('.', 1)[0], fromlist=[aug["class_name"].rsplit('.', 1)[-1]]),
                    aug["class_name"].rsplit('.', 1)[-1])(**aug["params"])
            for aug in cfg.augmentation.valid.augs
        ])

        valid_dataset = PatchClassificationDataset(
            valid_df,
            cfg.data.folder_path,
            transforms=valid_transforms
        )
        from inference import display_sample_predictions
        display_sample_predictions(detection_model, valid_dataset, num_samples=4)

        # Predict on test folder if provided
        if str(cfg.test.folder_path).lower() != "none":
            test_folder = cfg.test.folder_path
            output_excel = "/kaggle/working/test_predictions.xlsx"
            print("\n[Main] Predicting on test folder:")
            from inference import predict_test_folder
            predict_test_folder(
                detection_model,
                test_folder,
                valid_transforms,
                output_excel,
                print_results=True,
                model_path=pretrained_ckpt
            )

    print("[Main] Training finished. Best model is saved in:", best_model_folder)


if __name__ == "__main__":
    main()
