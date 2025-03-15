# Microorganism Binary Classification with PyTorch Lightning and Optuna

This repository contains code for training a microorganism detection/classification model using PyTorch Lightning. The code is organized into multiple modules for configuration, data handling, model building, custom callbacks, training, and inference.

## Project Structure

- **Configuration & Mode Settings**  
  The code supports different operational modes:
  - **Training Mode:**  
    - Run a complete training pipeline.
    - **Tuning Mode:** When enabled (`tuning_mode=True`), the model hyperparameters are tuned using [Optuna](https://optuna.org). In tuning mode, any provided pretrained checkpoint is ignored.
    - **Cross-Validation (CV):** You can enable or disable cross-validation. When enabled (by setting `use_cv=True`), training will run on multiple folds and aggregate evaluation metrics.
  - **Test Only Mode:**  
    - The code will load a pretrained checkpoint and run inference on a specified test folder.
    - It saves an Excel file with predictions (including predicted labels and probabilities) and produces evaluation plots such as the ROC curve and confusion matrix.

## Configuration Settings

The training pipeline is controlled by a configuration file (managed via OmegaConf) with the following key settings:

- **run_mode:** `"train"` or `"test"`  
  Controls whether the pipeline runs training or test-only inference.

- **tuning_mode:** `True` or `False`  
  If `True`, hyperparameter tuning via Optuna is activated and any provided pretrained checkpoint is ignored.

- **use_cv:** `True` or `False`  
  If `True`, cross-validation is used during training; otherwise, a single training fold is used.

- **use_optuna:** `True` or `False`  
  If `True`, the code uses Optuna for hyperparameter tuning; if `False`, manual parameter settings are used.

- **general:**  
  - **save_dir:** Base directory for saving all outputs (checkpoints, plots, evaluations).
  - **project_name:** Name identifier for the project (used for logging and naming outputs).

- **trainer:** Settings for the PyTorch Lightning Trainer, including number of devices, accelerator type, precision, and gradient clipping.

- **training:**  
  Contains parameters for:
  - **Random seed**
  - **tuning_epochs_detection:** Number of epochs for initial training/tuning.
  - **additional_epochs_detection:** Additional epochs for fine-tuning.
  - **cross_validation:** Whether to use CV.
  - **num_folds:** Number of folds (if CV is enabled).
  - **repeated_cv:** Number of repeated CV runs.
  - **composite_metric:** A dictionary with `alpha` and `beta` used to compute the composite metric  
    *Composite Metric = alpha * Validation Recall - beta * Validation Loss*

- **optimizer & scheduler:**  
  Specify the optimizer class and parameters (e.g., learning rate, weight decay) and scheduler settings.

- **model:**  
  Defines the backbone model configuration (e.g., ResNet50 with pretrained weights).

- **data:**  
  Specifies the CSV file with training data, image folder path, batch size, number of workers, validation split, etc.

- **augmentation:**  
  Contains augmentation settings for training and validation (using Albumentations).

- **test:**  
  Specifies the test folder path and optionally a CSV file with ground truth labels.

- **pretrained_ckpt:**  
  Path to a pretrained checkpoint (used only when `tuning_mode` is `False`).

- **optuna:**  
  Contains hyperparameter tuning settings:
  - **n_trials:** Number of Optuna trials.
  - **params:** Dictionary defining hyperparameters to tune (with type, range, and values).

## Evaluation Metrics

During evaluation, the following metrics and plots are generated and saved in the evaluation folder (`eval`):

- **Confusion Matrix:** Displays true vs. predicted labels.
- **Classification Report:** Includes F1 score, accuracy, precision, and recall.
- **Weighted F2 Score:** Computed with β = 2 (giving more weight to recall).
- **ROC Curve & AUC:** The ROC curve is plotted and the Area Under the Curve (AUC) is calculated.
- **Training Curves:** Loss and recall curves (per epoch) are plotted and saved in the `plots` folder.

## Tuning Mode (Optuna)

When tuning mode is active (`tuning_mode=True` and `use_optuna=True`):

- **Objective Function:**  
  The objective is defined using a composite metric:
  
  Composite Metric = alpha * Validation Recall - beta * Validation Loss
  
  where `alpha` and `beta` are configurable parameters.

- **Hyperparameter Tuning:**  
  Optuna runs multiple trials (as defined by `n_trials`) to search for the best hyperparameters.
  - **Trial Logging:** All trial information (parameters and outcomes) is saved to an Excel file (`optuna_trials.xlsx`) in the evaluation folder.
  - **Best Trial Logging:** The best trial’s parameters and its composite score are saved separately in `optuna_best_params.xlsx`.
  - **Visualizations:** Optuna visualizations (optimization history, parameter importance, and slice plots) are generated and saved in the evaluation folder.

## Final Outputs

At the end of a training run, you will receive:

- **Model Checkpoint:**  
  The best detection model is saved as a checkpoint file.

- **Training Curves:**  
  The loss and recall curves (from the final training run) are saved in `BASE_SAVE_DIR/plots`.

- **Evaluation Plots:**  
  Confusion matrix and ROC curve images are stored in the `eval` folder.

- **Optuna Logs & Visualizations:**  
  If tuning is used, Excel files with all trial details and best parameters, along with Optuna’s visualization images, are saved in `eval`.

- **Test Predictions (Test Mode):**  
  When running in test mode, an Excel file with predicted labels and probabilities (plus sample prediction images) is generated.

## Expected Results

- In **training mode**, the code logs training progress via training curves and prints evaluation metrics (including the composite metric based on recall and loss).  
- In **tuning mode**, the best trial is selected based on the composite metric, and only the best value is printed to the console while detailed parameters are logged in Excel.  
- In **test only mode**, the model loads a pretrained checkpoint, performs inference on the test dataset, and saves the predictions and evaluation plots.

Feel free to modify the configuration parameters in the configuration file to suit your experiments.

## Installation

Install the required packages with:

```bash
pip install torch torchvision pytorch-lightning optuna albumentations omegaconf scikit-learn matplotlib seaborn tqdm
