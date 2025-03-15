#!/usr/bin/env python
"""Dataset and data-related utilities."""

import os
import cv2
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Any, Tuple, Optional
from utils import load_obj


class PatchClassificationDataset(Dataset):
    """
    Custom dataset for patch classification.

    Args:
        data (Any): DataFrame or path to CSV file containing filenames/labels.
        image_dir (str): Path to the directory containing images.
        transforms (Any): Albumentations transforms.
        image_col (str): Column name for image filenames in the CSV/dataframe.
    """

    def __init__(
        self,
        data: Any,
        image_dir: str,
        transforms: Optional[Any] = None,
        image_col: str = "filename"
    ) -> None:
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data

        valid_rows = []
        for idx, row in df.iterrows():
            primary_path = os.path.join(image_dir, row[image_col])
            alternative_folder = image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row[image_col])
            if os.path.exists(primary_path) or os.path.exists(alternative_path):
                valid_rows.append(row)
            else:
                print(f"Warning: Image not found for row {idx}: {primary_path} or {alternative_path}. Skipping.")

        self.df = pd.DataFrame(valid_rows)
        self.image_dir = image_dir
        self.transforms = transforms
        self.image_col = image_col

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Get one sample of the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[Any, int]: (image tensor, label).
        """
        row = self.df.iloc[idx]
        primary_path = os.path.join(self.image_dir, row[self.image_col])
        if os.path.exists(primary_path):
            image_path = primary_path
        else:
            alternative_folder = self.image_dir.replace("Fa", "test")
            alternative_path = os.path.join(alternative_folder, row[self.image_col])
            if os.path.exists(alternative_path):
                image_path = alternative_path
                print(f"Using alternative image path: {image_path}")
            else:
                raise FileNotFoundError(f"Image not found: {primary_path} or {alternative_path}")

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        label = int(row["label"])
        return image, label


def preview_data_before_after(
    train_df: pd.DataFrame,
    folder_path: str,
    transform,
    n: int = 4
) -> None:
    """
    Preview a few samples before and after augmentation.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        folder_path (str): Path to the folder containing images.
        transform (Any): Albumentations transform for training.
        n (int): Number of samples to preview.
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
            # If needed, convert back to numpy for visualization
            image_aug = image_aug.permute(1, 2, 0).cpu().numpy()
            image_aug = image_aug.astype(np.uint8)

        axs[i, 1].imshow(image_aug)
        axs[i, 1].set_title(f"After Aug - Label: {row['label']}")
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("preview_augmentations.png")
    plt.close()
