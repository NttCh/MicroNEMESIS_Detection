#!/usr/bin/env python
"""Inference and evaluation utilities."""

import os
import cv2
import random
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, roc_curve, auc
from typing import Optional
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from data import PatchClassificationDataset
from utils import load_obj
from config import BASE_SAVE_DIR


def evaluate_test_roc(model: torch.nn.Module, csv_path: str, folder_path: str, transform) -> None:
    """
    Evaluate and plot the ROC curve for a model using a CSV file.

    Args:
        model (torch.nn.Module): The trained model.
        csv_path (str): Path to CSV file containing data for evaluation.
        folder_path (str): Folder path containing images.
        transform (Any): Albumentations transform for validation/test.
    """
    df = pd.read_csv(csv_path)
    dataset = PatchClassificationDataset(df, folder_path, transforms=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_probs = []
    all_labels = []
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()

    eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
    os.makedirs(eval_folder, exist_ok=True)
    roc_save_path = os.path.join(eval_folder, "roc_curve.png")
    plt.savefig(roc_save_path)
    plt.show()
    print(f"[ROC] Saved ROC curve to {roc_save_path}")


def evaluate_model(model, csv_path: str, cfg, stage: str) -> None:
    """
    Evaluate the model on a validation split and plot confusion matrix.

    Args:
        model (torch.nn.Module): The trained model.
        csv_path (str): Path to CSV file for validation data.
        cfg (Any): Configuration object.
        stage (str): A label for the evaluation stage (e.g. "Detection").
    """
    from sklearn.model_selection import train_test_split

    full_df = pd.read_csv(csv_path)
    _, valid_df = train_test_split(
        full_df,
        test_size=cfg.data.valid_split,
        random_state=cfg.training.seed,
        stratify=full_df[cfg.data.label_col]
    )

    valid_transforms = []
    for aug in cfg.augmentation.valid.augs:
        valid_transforms.append(load_obj(aug["class_name"])(**aug["params"]))

    valid_dataset = PatchClassificationDataset(
        valid_df,
        cfg.data.folder_path,
        transforms=lambda x: apply_compose(valid_transforms, x)
    )

    valid_loader = DataLoader(valid_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=cfg.data.num_workers)

    all_preds, all_labels = [], []
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
    os.makedirs(eval_folder, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {stage}")
    cm_save_path = os.path.join(eval_folder, "confusion_matrix.png")
    plt.savefig(cm_save_path)
    plt.show()
    print(f"[Evaluate] Saved confusion matrix plot to {cm_save_path}")

    print("Classification Report (F1 scores):")
    print(classification_report(all_labels, all_preds))

    f2 = fbeta_score(all_labels, all_preds, beta=2, average='weighted', zero_division=0)
    print(f"Weighted F2 Score: {f2:.4f}")

    # Evaluate and show ROC curve at the end
    evaluate_test_roc(model, cfg.data.detection_csv, cfg.test.folder_path, lambda x: apply_compose(valid_transforms, x))


def display_sample_predictions(model, dataset, num_samples: int = 4) -> None:
    """
    Display a few sample predictions from a dataset.

    Args:
        model (torch.nn.Module): The trained model.
        dataset (PatchClassificationDataset): Dataset to sample from.
        num_samples (int): Number of samples to display.
    """
    if len(dataset) == 0:
        print("Dataset is empty, cannot show sample predictions.")
        return

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images, true_labels, pred_labels = [], [], []

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            if not isinstance(image, torch.Tensor):
                import torch
                image = torch.tensor(image).permute(2, 0, 1)

            input_tensor = image.unsqueeze(0).to(device)
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()

            images.append(image.cpu().permute(1, 2, 0).numpy())
            true_labels.append(label)
            pred_labels.append(pred)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, num_samples, figsize=(5 * num_samples, 5))
    if num_samples == 1:
        axs = [axs]

    for i in range(num_samples):
        axs[i].imshow(images[i].astype(np.uint8))
        axs[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        axs[i].axis("off")

    eval_folder = os.path.join(BASE_SAVE_DIR, "eval")
    os.makedirs(eval_folder, exist_ok=True)
    sample_plot_path = os.path.join(eval_folder, "sample_predictions.png")
    plt.savefig(sample_plot_path)
    plt.show()
    print(f"[Display] Saved sample predictions plot to {sample_plot_path}")


def predict_test_folder(
    model: torch.nn.Module,
    test_folder: str,
    transform,
    output_excel: str,
    print_results: bool = True,
    model_path: Optional[str] = None
) -> None:
    """
    Predict on all images in a test folder and save results in an Excel file.

    Args:
        model (torch.nn.Module): The trained model.
        test_folder (str): Path to the test images folder.
        transform (Any): Albumentations transform for inference.
        output_excel (str): Path to save the Excel file with predictions.
        print_results (bool): Whether to print predictions.
        model_path (str): Optional path to a model checkpoint to load.
    """
    if not test_folder or test_folder.lower() == "none":
        print("No test folder provided. Skipping test predictions.")
        return

    if model_path and model_path.lower() != "none":
        print(f"Loading model checkpoint from {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith("model."):
                new_key = k[len("model."):]
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)

    image_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in test folder. Skipping test predictions.")
        return

    predictions = []
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        for file in image_files:
            image = cv2.imread(file, cv2.IMREAD_COLOR)
            if image is None:
                print(f"Warning: Could not read {file}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            augmented = transform(image=image)
            image_tensor = augmented["image"]
            if not isinstance(image_tensor, torch.Tensor):
                import torch
                image_tensor = torch.tensor(image_tensor)

            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(device)
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).item()
            prob = torch.softmax(logits, dim=1)[0, 1].item()

            predictions.append({"filename": file, "predicted_label": pred, "probability": prob})
            if print_results:
                print(f"File: {file} -> Predicted Label: {pred}, Prob: {prob:.8f}")

    predictions = sorted(predictions, key=lambda x: x["filename"])
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Predictions"
    ws.append(["Filename", "Predicted Label", "Probability", "Image"])

    row_num = 2
    for pred in predictions:
        ws.append([pred["filename"], pred["predicted_label"], pred["probability"]])
        try:
            img = XLImage(pred["filename"])
            img.width = 100
            img.height = 100
            cell_ref = f"D{row_num}"
            ws.add_image(img, cell_ref)
        except Exception as e:
            print(f"Could not insert image for {pred['filename']}: {e}")
        row_num += 1

    wb.save(output_excel)
    print(f"Saved test predictions with images to {output_excel}")

    pred_df = pd.DataFrame(predictions)
    print("\nFinal Test Predictions:")
    print(pred_df.to_string(index=False))


def apply_compose(augs, x):
    """
    Helper to apply a list of Albumentations transforms sequentially (manually),
    for times when you cannot directly create a Compose object.

    Args:
        augs (list): List of Albumentations transforms.
        x (dict): Dict with "image": <image>.

    Returns:
        dict: Dict with transformed "image".
    """
    image = x["image"]
    for aug in augs:
        image = aug(image=image)["image"]
    return {"image": image}
