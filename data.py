"""
Dataset definitions and data-related utilities.
"""

import os
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import albumentations as A
from typing import Any, Tuple, Optional
from omegaconf import DictConfig


class PatchClassificationDataset(Dataset):
    """
    A dataset class for patch classification. Loads images from a CSV file
    and applies transformations.
    """
    def __init__(
        self,
        data: Any,
        image_dir: str,
        transforms: Optional[A.Compose] = None,
        image_col: str = "filename"
    ) -> None:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data

        valid_rows = []
        for idx, row in df.iterrows():
            primary_path = os.path.join(image_dir, row[image_col])
            alt_folder = image_dir.replace("Fa", "test")
            alt_path = os.path.join(alt_folder, row[image_col])
            if os.path.exists(primary_path) or os.path.exists(alt_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image not found for row {idx}: {primary_path} or {alt_path}. Skipping.")

        self.df = pd.DataFrame(valid_rows)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_col = image_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        row = self.df.iloc[idx]
        primary_path = os.path.join(self.image_dir, row[self.image_col])

        if os.path.exists(primary_path):
            image_path = primary_path
        else:
            alt_folder = self.image_dir.replace("Fa", "test")
            alt_path = os.path.join(alt_folder, row[self.image_col])
            if os.path.exists(alt_path):
                image_path = alt_path
                print(f"Using alternative image path: {image_path}")
            else:
                raise FileNotFoundError(f"Image not found: {primary_path} or {alt_path}")

        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        label = int(row["label"])
        return image, label


def preview_data_before_after(
    train_df: pd.DataFrame,
    folder_path: str,
    transform: A.Compose,
    n: int = 4
) -> None:
    """
    Display sample images before and after augmentation.
    """
    indices = random.sample(range(len(train_df)), k=min(n, len(train_df)))
    fig, axs = plt.subplots(n, 2, figsize=(8, 4 * n))

    for i, idx in enumerate(indices):
        row = train_df.iloc[idx]
        image_path = os.path.join(folder_path, row["filename"])
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        axs[i, 0].imshow(image_rgb.astype(np.uint8))
        axs[i, 0].set_title(f"Before Aug - Label: {row['label']}")
        axs[i, 0].axis("off")

        augmented = transform(image=image_rgb)
        image_aug = augmented["image"]
        if isinstance(image_aug, torch.Tensor):
            image_aug = image_aug.permute(1, 2, 0).cpu().numpy()

        axs[i, 1].imshow(image_aug.astype(np.uint8))
        axs[i, 1].set_title(f"After Aug - Label: {row['label']}")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.show()
