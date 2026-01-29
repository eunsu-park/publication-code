"""
Dataset for Magnetogram to UV/EUV Image Translation.

Park et al. (2019), ApJL, 884, L23
https://doi.org/10.3847/2041-8213/ab46bb

This module provides dataset classes for loading magnetogram and EUV data.
- Input: SDO/HMI magnetogram
- Output: SDO/AIA EUV/UV images (1 or more wavelengths)
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Available wavelengths
WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]


def normalize_magnetogram(data: np.ndarray, mag_range: float = 1000.0) -> np.ndarray:
    """
    Normalize HMI magnetogram.

    Args:
        data: Raw magnetogram data.
        mag_range: Normalization factor (default: 1000.0).

    Returns:
        Normalized data.
    """
    return data / mag_range


def denormalize_magnetogram(data: np.ndarray, mag_range: float = 1000.0) -> np.ndarray:
    """
    Denormalize HMI magnetogram.

    Args:
        data: Normalized magnetogram data.
        mag_range: Normalization factor.

    Returns:
        Denormalized data.
    """
    return data * mag_range


def normalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Normalize AIA EUV/UV image.

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
    Denormalize AIA EUV/UV image.

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
    Base dataset for magnetogram-to-EUV translation.

    Loads .npz files containing HMI magnetogram and AIA EUV images.
    - Input: hmi_mag (1024x1024)
    - Output: aia_{wavelength} (1024x1024) x N channels

    Args:
        data_dir: Directory containing .npz files.
        wavelengths: List of target wavelengths.
        mag_range: Magnetogram normalization factor.
    """

    def __init__(
        self,
        data_dir: str,
        wavelengths: List[int],
        mag_range: float = 1000.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.wavelengths = wavelengths
        self.mag_range = mag_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Validate wavelengths
        for wl in wavelengths:
            if wl not in WAVELENGTHS:
                raise ValueError(
                    f"Invalid wavelength {wl}. "
                    f"Available: {WAVELENGTHS}"
                )

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npz file
        data = np.load(self.file_list[idx])

        # Input: HMI magnetogram
        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.mag_range)

        # Add channel dimension: (H, W) -> (1, H, W)
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        # Output: AIA EUV images (concatenated along channel dimension)
        aia_channels = []
        for wl in self.wavelengths:
            aia = data[f"aia_{wl}"].astype(np.float32)
            aia = normalize_euv(aia)
            if aia.ndim == 2:
                aia = aia[np.newaxis, ...]
            aia_channels.append(aia)

        # Concatenate channels: (N, H, W)
        aia_stack = np.concatenate(aia_channels, axis=0)

        return (
            torch.from_numpy(hmi_mag).float(),
            torch.from_numpy(aia_stack).float(),
        )


class TrainDataset(BaseDataset):
    """Training dataset for magnetogram-to-EUV translation."""

    pass


class ValidationDataset(BaseDataset):
    """Validation dataset for magnetogram-to-EUV translation."""

    pass


class TestDataset(BaseDataset):
    """Test dataset for magnetogram-to-EUV translation."""

    pass
