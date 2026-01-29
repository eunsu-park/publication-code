"""
Validation Script for Magnetogram to UV/EUV Image Translation.

Park et al. (2019), ApJL, 884, L23
https://doi.org/10.3847/2041-8213/ab46bb

This script loads a trained checkpoint and evaluates on the validation set.

Usage:
    python validation.py --checkpoint ./checkpoints/checkpoint_best.pth
    python validation.py --checkpoint ./checkpoints/checkpoint_best.pth --data_dir ./data
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import ValidationDataset
from networks import Generator


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    wavelengths: List[int],
) -> Dict[str, float]:
    """
    Run evaluation on validation set.

    Args:
        model: Generator model.
        val_loader: Validation data loader.
        device: Device to run evaluation on.
        wavelengths: List of target wavelengths.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    criterion_l1 = nn.L1Loss(reduction='none')
    criterion_mse = nn.MSELoss(reduction='none')

    # Per-channel metrics
    channel_l1 = {wl: 0.0 for wl in wavelengths}
    channel_mse = {wl: 0.0 for wl in wavelengths}
    total_samples = 0

    with torch.no_grad():
        for magnetogram, euv in val_loader:
            magnetogram = magnetogram.to(device)
            euv = euv.to(device)

            output = model(magnetogram)
            batch_size = magnetogram.size(0)

            # Compute per-channel metrics
            for i, wl in enumerate(wavelengths):
                l1 = criterion_l1(output[:, i:i+1], euv[:, i:i+1]).mean()
                mse = criterion_mse(output[:, i:i+1], euv[:, i:i+1]).mean()
                channel_l1[wl] += l1.item() * batch_size
                channel_mse[wl] += mse.item() * batch_size

            total_samples += batch_size

    # Compute averages
    metrics = {}
    total_l1 = 0.0
    total_mse = 0.0

    for wl in wavelengths:
        avg_l1 = channel_l1[wl] / total_samples
        avg_mse = channel_mse[wl] / total_samples
        metrics[f"l1_{wl}"] = avg_l1
        metrics[f"mse_{wl}"] = avg_mse
        metrics[f"rmse_{wl}"] = np.sqrt(avg_mse)
        total_l1 += avg_l1
        total_mse += avg_mse

    # Overall metrics
    metrics["l1_loss"] = total_l1 / len(wavelengths)
    metrics["mse_loss"] = total_mse / len(wavelengths)
    metrics["rmse"] = np.sqrt(metrics["mse_loss"])
    metrics["num_samples"] = total_samples

    return metrics


def main() -> None:
    """Main validation function."""
    # Load config
    config = TrainConfig.from_args()

    # Check checkpoint path
    checkpoint_path = Path(config.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint
    device = torch.device(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get wavelengths from checkpoint or config
    wavelengths = checkpoint.get("wavelengths", config.wavelength_list)

    # Create model
    generator = Generator(
        in_channels=config.in_channels,
        out_channels=len(wavelengths),
        base_features=config.ngf,
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Wavelengths: {wavelengths}")

    # Create validation dataset
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        wavelengths=wavelengths,
        mag_range=config.mag_range,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(f"Validation samples: {len(val_dataset)}")
    print(f"Device: {device}")
    print("-" * 50)

    # Run evaluation
    metrics = evaluate(generator, val_loader, device, wavelengths)

    # Print results
    print("Validation Results:")
    print(f"  Overall L1 Loss:  {metrics['l1_loss']:.6f}")
    print(f"  Overall MSE Loss: {metrics['mse_loss']:.6f}")
    print(f"  Overall RMSE:     {metrics['rmse']:.6f}")
    print()
    print("  Per-channel metrics:")
    for wl in wavelengths:
        print(f"    {wl}A: L1={metrics[f'l1_{wl}']:.6f}, RMSE={metrics[f'rmse_{wl}']:.6f}")
    print(f"  Samples: {metrics['num_samples']}")
    print("-" * 50)

    # Save results to file
    output_dir = Path(config.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "validation_results.txt"

    with open(result_path, "w") as f:
        f.write("Validation Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Wavelengths: {wavelengths}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall L1 Loss:  {metrics['l1_loss']:.6f}\n")
        f.write(f"Overall MSE Loss: {metrics['mse_loss']:.6f}\n")
        f.write(f"Overall RMSE:     {metrics['rmse']:.6f}\n")
        f.write("-" * 40 + "\n")
        f.write("Per-channel metrics:\n")
        for wl in wavelengths:
            f.write(f"  {wl}A: L1={metrics[f'l1_{wl}']:.6f}, RMSE={metrics[f'rmse_{wl}']:.6f}\n")

    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
