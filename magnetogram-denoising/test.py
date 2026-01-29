"""
Test Script for Magnetogram Denoising.

Park et al. (2020), ApJL, 891, L4
https://doi.org/10.3847/2041-8213/ab74d2

This script loads a trained checkpoint, evaluates on the test set,
and saves predictions in npz format.

Usage:
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results --data_dir ./data
"""

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import InferenceConfig
from dataset import TestDataset, denormalize
from networks import Generator


def compute_metrics(
    output: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        output: Predicted output.
        target: Ground truth target.

    Returns:
        Dictionary of metrics.
    """
    # L1 (Mean Absolute Error)
    mae = np.mean(np.abs(output - target))

    # MSE
    mse = np.mean((output - target) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # Pixel-wise correlation coefficient
    output_flat = output.flatten()
    target_flat = target.flatten()
    corr = np.corrcoef(output_flat, target_flat)[0, 1]

    # Peak Signal-to-Noise Ratio (PSNR)
    data_range = target.max() - target.min()
    if data_range > 0:
        psnr = 10 * np.log10((data_range ** 2) / mse) if mse > 0 else float("inf")
    else:
        psnr = float("inf")

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "correlation": float(corr),
        "psnr": float(psnr),
    }


def test_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    data_range: float = 100.0,
) -> Dict[str, float]:
    """
    Run inference on test set and save predictions.

    Args:
        model: Generator model.
        test_loader: Test data loader.
        output_dir: Directory to save results.
        device: Device to run inference on.
        data_range: Normalization factor for denormalization.

    Returns:
        Dictionary of aggregated metrics.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for idx, (noisy, target) in enumerate(test_loader):
            noisy = noisy.to(device)

            # Generate output
            output = model(noisy)

            # Move to CPU and numpy
            noisy_np = noisy.cpu().numpy()
            target_np = target.numpy()
            output_np = output.cpu().numpy()

            # Process each sample in batch
            for b in range(noisy_np.shape[0]):
                sample_idx = idx * test_loader.batch_size + b

                # Get single sample
                input_sample = noisy_np[b]    # (1, H, W)
                target_sample = target_np[b]  # (1, H, W)
                output_sample = output_np[b]  # (1, H, W)

                # Denormalize
                input_denorm = denormalize(input_sample, data_range)
                target_denorm = denormalize(target_sample, data_range)
                output_denorm = denormalize(output_sample, data_range)

                # Compute metrics (on denormalized data)
                metrics = compute_metrics(output_denorm, target_denorm)
                metrics["sample_idx"] = sample_idx
                all_metrics.append(metrics)

                # Save as npz
                save_path = output_dir / f"result_{sample_idx:04d}.npz"
                np.savez(
                    save_path,
                    input=input_denorm,
                    target=target_denorm,
                    prediction=output_denorm,
                )

    # Compute aggregated metrics
    agg_metrics = {
        "mae": np.mean([m["mae"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "rmse": np.mean([m["rmse"] for m in all_metrics]),
        "correlation": np.mean([m["correlation"] for m in all_metrics]),
        "psnr": np.mean([m["psnr"] for m in all_metrics if m["psnr"] != float("inf")]),
        "num_samples": len(all_metrics),
    }

    # Save per-sample metrics to CSV
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)

    return agg_metrics


def main() -> None:
    """Main test function."""
    # Load config
    config = InferenceConfig.from_args()

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

    # Create test dataset
    test_dataset = TestDataset(
        data_dir=f"{config.data_dir}/test",
        input_size=config.input_size,
        data_range=config.data_range,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Use single worker for ordered saving
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")
    print("-" * 50)

    # Run test and save
    output_dir = Path(config.output_path)
    metrics = test_and_save(
        generator,
        test_loader,
        output_dir,
        device,
        data_range=config.data_range,
    )

    # Print results
    print("Test Results:")
    print(f"  MAE:         {metrics['mae']:.6f}")
    print(f"  MSE:         {metrics['mse']:.6f}")
    print(f"  RMSE:        {metrics['rmse']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")
    print(f"  PSNR:        {metrics['psnr']:.2f} dB")
    print(f"  Samples:     {metrics['num_samples']}")
    print("-" * 50)
    print(f"Results saved to: {output_dir}")
    print(f"Metrics CSV: {output_dir / 'metrics.csv'}")

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Test Results Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"MAE:         {metrics['mae']:.6f}\n")
        f.write(f"MSE:         {metrics['mse']:.6f}\n")
        f.write(f"RMSE:        {metrics['rmse']:.6f}\n")
        f.write(f"Correlation: {metrics['correlation']:.6f}\n")
        f.write(f"PSNR:        {metrics['psnr']:.2f} dB\n")


if __name__ == "__main__":
    main()
