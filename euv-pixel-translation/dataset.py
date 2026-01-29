"""
Dataset for EUV Pixel-to-Pixel Translation.

Park et al. (2023), ApJS, 264, 33
https://doi.org/10.3847/1538-4365/aca902

This module provides dataset classes for loading EUV images.
- Input: SDO/AIA EUV images (configurable wavelengths, default: 171, 193, 211 nm)
- Output: SDO/AIA EUV images (configurable wavelengths, default: 94, 131, 335 nm)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Available wavelengths
WAVELENGTHS = [94, 131, 171, 193, 211, 335]


def normalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Normalize AIA EUV image.

    Args:
        data: Raw EUV data.

    Returns:
        Normalized data in approximately [-1, 1] range.
    """
    data = np.clip(data + 1, 1, None)
    data = np.log2(data)
    data = (data / 7) - 1.0
    return data


def denormalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Denormalize AIA EUV image.

    Args:
        data: Normalized EUV data.

    Returns:
        Denormalized data.
    """
    data = (data + 1.0) * 7
    data = np.power(2, data) - 1
    return data


class BaseDataset(Dataset):
    """
    Base dataset for EUV pixel-to-pixel translation.

    Loads .npz files containing AIA EUV images.
    - Input: selected input wavelengths
    - Output: selected output wavelengths

    Args:
        data_dir: Directory containing .npz files.
        input_wavelengths: List of input wavelengths.
        output_wavelengths: List of output wavelengths.
        model_type: Model type ("fcn" or "cnn") for format selection.
    """

    def __init__(
        self,
        data_dir: str,
        input_wavelengths: List[int],
        output_wavelengths: List[int],
        model_type: str = "fcn",
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.input_wavelengths = input_wavelengths
        self.output_wavelengths = output_wavelengths
        self.model_type = model_type.lower()
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Validate wavelengths
        for wl in input_wavelengths + output_wavelengths:
            if wl not in WAVELENGTHS:
                raise ValueError(
                    f"Invalid wavelength {wl}. Available: {WAVELENGTHS}"
                )

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npz file
        data = np.load(self.file_list[idx])

        # Load and normalize input channels
        input_channels = []
        for wl in self.input_wavelengths:
            img = data[f"aia_{wl}"].astype(np.float32)
            img = normalize_euv(img)
            input_channels.append(img)

        # Load and normalize output channels
        output_channels = []
        for wl in self.output_wavelengths:
            img = data[f"aia_{wl}"].astype(np.float32)
            img = normalize_euv(img)
            output_channels.append(img)

        # Stack channels
        input_stack = np.stack(input_channels, axis=-1)   # (H, W, C)
        output_stack = np.stack(output_channels, axis=-1)  # (H, W, C)

        # Convert format based on model type
        if self.model_type == "cnn":
            # CNN expects (C, H, W)
            input_stack = np.transpose(input_stack, (2, 0, 1))
            output_stack = np.transpose(output_stack, (2, 0, 1))

        return (
            torch.from_numpy(input_stack).float(),
            torch.from_numpy(output_stack).float(),
        )


class TrainDataset(BaseDataset):
    """Training dataset for EUV pixel-to-pixel translation."""

    pass


class ValidationDataset(BaseDataset):
    """Validation dataset for EUV pixel-to-pixel translation."""

    pass


class TestDataset(BaseDataset):
    """Test dataset for EUV pixel-to-pixel translation."""

    pass
