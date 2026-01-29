"""
Dataset for Far-side Magnetogram Generation.

Kim, Park et al. (2019), Nature Astronomy, 3, 397
https://doi.org/10.1038/s41550-019-0711-5

This module provides dataset classes for loading EUV and magnetogram data.
- Train/Valid Input: SDO/AIA 304 nm
- Train/Valid Target: SDO/HMI magnetogram
- Test Input: STEREO/EUVI 304 nm (far-side)
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def normalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Normalize EUV 304 nm image.

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
    Denormalize EUV data.

    Args:
        data: Normalized EUV data.

    Returns:
        Denormalized data.
    """
    data = (data + 1.0) * 7
    data = np.power(2, data) - 1
    return data


def normalize_magnetogram(data: np.ndarray, data_range: float = 100.0) -> np.ndarray:
    """
    Normalize magnetogram.

    Args:
        data: Raw magnetogram data.
        data_range: Normalization factor.

    Returns:
        Normalized data.
    """
    return data / data_range


def denormalize_magnetogram(data: np.ndarray, data_range: float = 100.0) -> np.ndarray:
    """
    Denormalize magnetogram.

    Args:
        data: Normalized magnetogram data.
        data_range: Normalization factor.

    Returns:
        Denormalized data.
    """
    return data * data_range


class TrainDataset(Dataset):
    """
    Training dataset for far-side magnetogram generation.

    Loads .npz files containing AIA 304 nm and HMI magnetogram pairs.
    - Input: aia_304 (1024x1024)
    - Target: hmi_mag (1024x1024)

    Args:
        data_dir: Directory containing .npz files.
        data_range: Magnetogram normalization factor (default: 100.0).
    """

    def __init__(
        self,
        data_dir: str,
        data_range: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npz file
        data = np.load(self.file_list[idx])

        # Input: AIA 304 nm
        aia_304 = data["aia_304"].astype(np.float32)
        aia_304 = normalize_euv(aia_304)

        # Target: HMI magnetogram
        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.data_range)

        # Add channel dimension: (H, W) -> (1, H, W)
        if aia_304.ndim == 2:
            aia_304 = aia_304[np.newaxis, ...]
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        return (
            torch.from_numpy(aia_304).float(),
            torch.from_numpy(hmi_mag).float(),
        )


class ValidationDataset(Dataset):
    """
    Validation dataset for far-side magnetogram generation.

    Same format as TrainDataset.
    """

    def __init__(
        self,
        data_dir: str,
        data_range: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.file_list[idx])

        aia_304 = data["aia_304"].astype(np.float32)
        aia_304 = normalize_euv(aia_304)

        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.data_range)

        if aia_304.ndim == 2:
            aia_304 = aia_304[np.newaxis, ...]
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        return (
            torch.from_numpy(aia_304).float(),
            torch.from_numpy(hmi_mag).float(),
        )


class TestDataset(Dataset):
    """
    Test dataset for far-side magnetogram generation.

    For near-side test: loads .npz files with aia_304 and hmi_mag.
    For far-side inference: loads .npz files with euvi_304 only.

    Args:
        data_dir: Directory containing .npz files.
        data_range: Magnetogram normalization factor (default: 100.0).
        farside: If True, loads EUVI 304 nm (no target).
    """

    def __init__(
        self,
        data_dir: str,
        data_range: float = 100.0,
        farside: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.farside = farside
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def get_filepath(self, idx: int) -> Path:
        """Get the file path for a given index."""
        return self.file_list[idx]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        filepath = self.file_list[idx]
        data = np.load(filepath)

        if self.farside:
            # Far-side: EUVI 304 nm (no target)
            euvi_304 = data["euvi_304"].astype(np.float32)
            euvi_304 = normalize_euv(euvi_304)

            if euvi_304.ndim == 2:
                euvi_304 = euvi_304[np.newaxis, ...]

            # Return empty target
            target = torch.zeros_like(torch.from_numpy(euvi_304))
            return torch.from_numpy(euvi_304).float(), target
        else:
            # Near-side: AIA 304 nm with HMI magnetogram target
            aia_304 = data["aia_304"].astype(np.float32)
            aia_304 = normalize_euv(aia_304)

            hmi_mag = data["hmi_mag"].astype(np.float32)
            hmi_mag = normalize_magnetogram(hmi_mag, self.data_range)

            if aia_304.ndim == 2:
                aia_304 = aia_304[np.newaxis, ...]
            if hmi_mag.ndim == 2:
                hmi_mag = hmi_mag[np.newaxis, ...]

            return (
                torch.from_numpy(aia_304).float(),
                torch.from_numpy(hmi_mag).float(),
            )
