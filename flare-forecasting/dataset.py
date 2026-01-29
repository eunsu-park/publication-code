"""
Dataset for Solar Flare Forecasting.

Park et al. (2018), ApJ, 869, 91
https://doi.org/10.3847/1538-4357/aaed40

This module provides dataset classes for loading magnetogram data.
- Input: HMI full-disk magnetogram (1024x1024)
- Output: Binary classification (Flare / No-flare)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Flare class folders
FLARE_CLASSES = ["no", "c", "m", "x"]


def get_positive_classes(threshold: str) -> List[str]:
    """
    Get positive (flare) classes based on threshold.

    Args:
        threshold: Flare threshold ("c", "m", or "x").

    Returns:
        List of class names that are considered positive (flare).
    """
    if threshold.lower() == "c":
        return ["c", "m", "x"]
    elif threshold.lower() == "m":
        return ["m", "x"]
    elif threshold.lower() == "x":
        return ["x"]
    else:
        raise ValueError(f"Invalid threshold: {threshold}. Must be 'c', 'm', or 'x'.")


class BaseDataset(Dataset):
    """
    Base dataset for flare forecasting.

    Loads .npy files from class-specific folders.
    - Folders: no/, c/, m/, x/
    - Binary label based on flare_threshold

    Args:
        data_dir: Directory containing class folders.
        flare_threshold: Threshold for binary classification.
        mag_range: Normalization range for magnetogram.
        augmentation: Whether to apply data augmentation.
        random_flip: Whether to apply random flipping.
        random_rotation: Whether to apply random rotation.
    """

    def __init__(
        self,
        data_dir: str,
        flare_threshold: str = "c",
        mag_range: float = 1000.0,
        augmentation: bool = False,
        random_flip: bool = True,
        random_rotation: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.flare_threshold = flare_threshold
        self.mag_range = mag_range
        self.augmentation = augmentation
        self.random_flip = random_flip
        self.random_rotation = random_rotation

        # Get positive classes based on threshold
        self.positive_classes = get_positive_classes(flare_threshold)

        # Collect all files with labels
        self.file_list: List[Tuple[Path, int]] = []
        self.class_counts = {"negative": 0, "positive": 0}

        for class_name in FLARE_CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            # Determine label (1: flare, 0: no flare)
            label = 1 if class_name in self.positive_classes else 0

            # Collect files
            for file_path in sorted(class_dir.glob("*.npy")):
                self.file_list.append((file_path, label))
                if label == 1:
                    self.class_counts["positive"] += 1
                else:
                    self.class_counts["negative"] += 1

        if len(self.file_list) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx][0]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for imbalanced data.

        Returns:
            Tensor of class weights [weight_negative, weight_positive].
        """
        total = self.class_counts["negative"] + self.class_counts["positive"]
        if total == 0:
            return torch.tensor([1.0, 1.0])

        weight_neg = total / (2 * self.class_counts["negative"]) if self.class_counts["negative"] > 0 else 1.0
        weight_pos = total / (2 * self.class_counts["positive"]) if self.class_counts["positive"] > 0 else 1.0

        return torch.tensor([weight_neg, weight_pos])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path, label = self.file_list[idx]

        # Load data
        data = np.load(file_path).astype(np.float32)

        # Normalize
        data = data / self.mag_range

        # Data augmentation (training only)
        if self.augmentation:
            # Random horizontal flip
            if self.random_flip and np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()

            # Random vertical flip
            if self.random_flip and np.random.random() > 0.5:
                data = np.flip(data, axis=0).copy()

            # Random 90-degree rotation
            if self.random_rotation:
                k = np.random.randint(0, 4)
                data = np.rot90(data, k).copy()

        # Add channel dimension: (H, W) -> (1, H, W)
        data = data[np.newaxis, ...]

        return torch.from_numpy(data).float(), label


class TrainDataset(BaseDataset):
    """Training dataset with augmentation enabled."""

    def __init__(
        self,
        data_dir: str,
        flare_threshold: str = "c",
        mag_range: float = 1000.0,
        random_flip: bool = True,
        random_rotation: bool = True,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            flare_threshold=flare_threshold,
            mag_range=mag_range,
            augmentation=True,
            random_flip=random_flip,
            random_rotation=random_rotation,
        )


class ValidationDataset(BaseDataset):
    """Validation dataset without augmentation."""

    def __init__(
        self,
        data_dir: str,
        flare_threshold: str = "c",
        mag_range: float = 1000.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            flare_threshold=flare_threshold,
            mag_range=mag_range,
            augmentation=False,
        )


class TestDataset(BaseDataset):
    """Test dataset without augmentation."""

    def __init__(
        self,
        data_dir: str,
        flare_threshold: str = "c",
        mag_range: float = 1000.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            flare_threshold=flare_threshold,
            mag_range=mag_range,
            augmentation=False,
        )
