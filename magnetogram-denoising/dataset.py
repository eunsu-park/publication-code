"""
Dataset for Magnetogram Denoising.

Park et al. (2020), ApJL, 891, L4
https://doi.org/10.3847/2041-8213/ab74d2

This module provides dataset classes for loading SDO/HMI magnetogram data.
- Input: Single noisy magnetogram (center frame)
- Target: 21-frame stacked (averaged) magnetogram
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """
    Base dataset for magnetogram denoising.

    Loads .npy files containing 21-frame magnetogram stacks.
    - Original shape: (21, 512, 512)
    - Center crop to: (21, 256, 256)
    - Input: center frame (index 10)
    - Target: mean of all 21 frames

    Args:
        data_dir: Directory containing .npy files.
        input_size: Size to crop (default: 256).
        data_range: Normalization factor (default: 100.0).
    """

    def __init__(
        self,
        data_dir: str,
        input_size: int = 256,
        data_range: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.data_range = data_range
        self.file_list = sorted(self.data_dir.glob("*.npy"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npy files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def _center_crop(self, data: np.ndarray) -> np.ndarray:
        """Center crop from 512x512 to input_size x input_size."""
        _, h, w = data.shape
        start_h = (h - self.input_size) // 2
        start_w = (w - self.input_size) // 2
        return data[
            :,
            start_h : start_h + self.input_size,
            start_w : start_w + self.input_size,
        ]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npy file: (21, 512, 512)
        data = np.load(self.file_list[idx])

        # Center crop: (21, 256, 256)
        data = self._center_crop(data)

        # Normalize
        data = data / self.data_range

        # Input: center frame (index 10)
        input_frame = data[10:11]  # Keep dim: (1, 256, 256)

        # Target: mean of all 21 frames
        target_frame = data.mean(axis=0, keepdims=True)  # (1, 256, 256)

        return (
            torch.from_numpy(input_frame).float(),
            torch.from_numpy(target_frame).float(),
        )

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]


class TrainDataset(BaseDataset):
    """Training dataset for magnetogram denoising."""

    pass


class ValidationDataset(BaseDataset):
    """Validation dataset for magnetogram denoising."""

    pass


class TestDataset(BaseDataset):
    """Test dataset for magnetogram denoising."""

    pass


def denormalize(data: np.ndarray, data_range: float = 100.0) -> np.ndarray:
    """
    Denormalize magnetogram data.

    Args:
        data: Normalized data.
        data_range: Normalization factor used during preprocessing.

    Returns:
        Denormalized data.
    """
    return data * data_range
