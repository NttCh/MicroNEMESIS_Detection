"""
Inference and evaluation-related functions.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score
from torch.utils.data import Dataset
import albumentations as A


def evaluate_model(model, csv_path: str, cfg, stage: str) -> None:
    """
    Evaluate the model on a validation set and display a confusion matrix,
    classification report, and F2 score.
    """
    from sklearn.model_selection import train_test_split
    from data import PatchClassificationDataset

    full_df = pd.read_csv(csv_path)
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

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers
    )

    all_preds = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix: {stage}")

    eval_folder = os.path.join(cfg.general.save_dir, "eval")
    os.makedirs(eval_folder, exist_ok=True)
    cm_save_path = os.path.join(eval_folder, "confusion_matrix.png")
    plt.savefig(cm_save_path)
    print(f"[Evaluate] Saved confusion matrix plot to {cm_save_path}")
    plt.show()

    print("Classification Report (F1 scores):")
    print(classification_report(all_labels, all_preds))

    f2_value = fbeta_score(all_labels, all_preds, beta=2, average='weighted', zero_division=0)
    print(f"Weighted F2 Score: {f2_value:.4f}")


def display_sample_predictions(model, dataset: Dataset, num_samples: int = 4) -> None:
    """
    Display a few sample predictions from a given dataset.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    if len(dataset) == 0:
        print("Dataset is empty, cannot show sample predictions.")
        return

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = []
    true_labels = []
    pred_labels = []

    model.eval()
    with torch.no_grad():
        for idx in indices:
            image, label = dataset[idx]
            if not isinstance(image, torch.Tensor):
                image = torch.tensor(image).permute(2, 0, 1)
            input_tensor = image.unsqueeze(0).to(model.device)
            logits = model(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            images.append(image.cpu().permute(1, 2, 0).numpy())
            true_labels.append(label)
            pred_labels.append(pred)

    model.train()
    fig, axs = plt.subplots(1, num_samples, figsize=(15, 5))

    for i in range(num_samples):
        axs[i].imshow(images[i].astype(np.uint8))
        axs[i].set_title(f"True: {true_labels[i]}\nPred: {pred_labels[i]}")
        axs[i].axis("off")

    eval_folder = os.path.join("baclogs3", "eval")
    os.makedirs(eval_folder, exist_ok=True)
    sample_plot_path = os.path.join(eval_folder, "sample_predictions.png")
    plt.savefig(sample_plot_path)
    print(f"[Display] Saved sample predictions plot to {sample_plot_path}")
    plt.show()


def predict_test_folder(
    model,
    test_folder: str,
    transform: A.Compose,
    output_excel: str,
    print_results: bool = True,
    model_path: Optional[str] = None
) -> None:
    """
    Predict on a test folder of images and save results (with embedded thumbnails)
    to an Excel file.
    """
    import pandas as pd
    import os
    import torch
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage

    if test_folder is None or str(test_folder).lower() == "none":
        print("No test folder provided. Skipping test predictions.")
        return

    if model_path is not None and model_path.lower() != "none":
        print(f"Loading model checkpoint from {model_path}")
        state_dict = torch.load(model_path, map_location=model.device)
        model.load_state_dict(state_dict)

    image_files = []
    for root, _, files in os.walk(test_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in test folder. Skipping test predictions.")
        return

    predictions = []
    model.eval()

    with torch.no_grad():
        for file in image_files:
            image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
            if image_bgr is None:
                print(f"Warning: Could not read {file}")
                continue
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
            augmented = transform(image=image)
            image_tensor = augmented["image"]

            if not isinstance(image_tensor, torch.Tensor):
                image_tensor = torch.tensor(image_tensor)

            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            image_tensor = image_tensor.to(model.device)
            logits = model(image_tensor)
            pred = torch.argmax(logits, dim=1).item()

            predictions.append({"filename": file, "predicted_label": pred})
            if print_results:
                print(f"File: {file} -> Predicted Label: {pred}")

    predictions = sorted(predictions, key=lambda x: x["filename"])
    wb = Workbook()
    ws = wb.active
    ws.title = "Test Predictions"
    ws.append(["Filename", "Predicted Label", "Image"])

    row_num = 2
    for pred in predictions:
        ws.append([pred["filename"], pred["predicted_label"]])
        try:
            img = XLImage(pred["filename"])
            img.width = 100
            img.height = 100
            cell_ref = f"C{row_num}"
            ws.add_image(img, cell_ref)
        except Exception as e:
            print(f"Could not insert image for {pred['filename']}: {e}")
        row_num += 1

    wb.save(output_excel)
    print(f"Saved test predictions with images to {output_excel}")

    pred_df = pd.DataFrame(predictions)
    print("\nFinal Test Predictions:")
    print(pred_df.to_string(index=False))
