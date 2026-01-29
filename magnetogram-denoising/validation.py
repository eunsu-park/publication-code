"""
Validation Script for Magnetogram Denoising.

Park et al. (2020), ApJL, 891, L4
https://doi.org/10.3847/2041-8213/ab74d2

This script loads a trained checkpoint and evaluates on the validation set.

Usage:
    python validation.py --checkpoint ./checkpoints/checkpoint_best.pth
    python validation.py --checkpoint ./checkpoints/checkpoint_best.pth --data_dir ./data
"""

from pathlib import Path
from typing import Dict

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
) -> Dict[str, float]:
    """
    Run evaluation on validation set.

    Args:
        model: Generator model.
        val_loader: Validation data loader.
        device: Device to run evaluation on.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    total_l1 = 0.0
    total_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for noisy, target in val_loader:
            noisy = noisy.to(device)
            target = target.to(device)

            output = model(noisy)

            # L1 loss
            loss_l1 = criterion_l1(output, target)
            total_l1 += loss_l1.item() * noisy.size(0)

            # MSE loss
            loss_mse = criterion_mse(output, target)
            total_mse += loss_mse.item() * noisy.size(0)

            total_samples += noisy.size(0)

    avg_l1 = total_l1 / total_samples
    avg_mse = total_mse / total_samples
    avg_rmse = np.sqrt(avg_mse)

    return {
        "l1_loss": avg_l1,
        "mse_loss": avg_mse,
        "rmse": avg_rmse,
        "num_samples": total_samples,
    }


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

    # Create model
    generator = Generator(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_features=config.ngf,
    )
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator = generator.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")

    # Create validation dataset
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        input_size=config.input_size,
        data_range=config.data_range,
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
    metrics = evaluate(generator, val_loader, device)

    # Print results
    print("Validation Results:")
    print(f"  L1 Loss:  {metrics['l1_loss']:.6f}")
    print(f"  MSE Loss: {metrics['mse_loss']:.6f}")
    print(f"  RMSE:     {metrics['rmse']:.6f}")
    print(f"  Samples:  {metrics['num_samples']}")
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
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"L1 Loss:  {metrics['l1_loss']:.6f}\n")
        f.write(f"MSE Loss: {metrics['mse_loss']:.6f}\n")
        f.write(f"RMSE:     {metrics['rmse']:.6f}\n")

    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
