"""
Test Script for Magnetogram to UV/EUV Image Translation.

Park et al. (2019), ApJL, 884, L23
https://doi.org/10.3847/2041-8213/ab46bb

This script loads a trained checkpoint, evaluates on the test set,
and saves predictions in npz format.

Usage:
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results --data_dir ./data
"""

import csv
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import InferenceConfig
from dataset import TestDataset, denormalize_euv, denormalize_magnetogram
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

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "correlation": float(corr),
    }


def test_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
    wavelengths: List[int],
    mag_range: float = 1000.0,
) -> Dict[str, float]:
    """
    Run inference on test set and save predictions.

    Args:
        model: Generator model.
        test_loader: Test data loader.
        output_dir: Directory to save results.
        device: Device to run inference on.
        wavelengths: List of target wavelengths.
        mag_range: Magnetogram normalization factor.

    Returns:
        Dictionary of aggregated metrics.
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-channel metrics
    all_metrics = {wl: [] for wl in wavelengths}
    overall_metrics = []

    with torch.no_grad():
        for idx, (magnetogram, euv) in enumerate(test_loader):
            magnetogram = magnetogram.to(device)

            # Generate output
            output = model(magnetogram)

            # Move to CPU and numpy
            mag_np = magnetogram.cpu().numpy()
            euv_np = euv.numpy()
            output_np = output.cpu().numpy()

            # Process each sample in batch
            for b in range(mag_np.shape[0]):
                sample_idx = idx * test_loader.batch_size + b

                # Get single sample
                input_sample = mag_np[b]      # (1, H, W)
                target_sample = euv_np[b]     # (N, H, W)
                output_sample = output_np[b]  # (N, H, W)

                # Denormalize
                input_denorm = denormalize_magnetogram(input_sample, mag_range)
                target_denorm = denormalize_euv(target_sample)
                output_denorm = denormalize_euv(output_sample)

                # Compute per-channel metrics
                sample_metrics = {"sample_idx": sample_idx}
                for i, wl in enumerate(wavelengths):
                    ch_metrics = compute_metrics(
                        output_denorm[i:i+1],
                        target_denorm[i:i+1]
                    )
                    sample_metrics[f"mae_{wl}"] = ch_metrics["mae"]
                    sample_metrics[f"corr_{wl}"] = ch_metrics["correlation"]
                    all_metrics[wl].append(ch_metrics)

                # Compute overall metrics
                overall = compute_metrics(output_denorm, target_denorm)
                sample_metrics["mae_overall"] = overall["mae"]
                sample_metrics["corr_overall"] = overall["correlation"]
                overall_metrics.append(sample_metrics)

                # Save as npz with per-channel data
                save_dict = {
                    "input": input_denorm,
                    "prediction": output_denorm,
                    "target": target_denorm,
                }
                # Also save per-channel for convenience
                for i, wl in enumerate(wavelengths):
                    save_dict[f"target_{wl}"] = target_denorm[i:i+1]
                    save_dict[f"prediction_{wl}"] = output_denorm[i:i+1]

                save_path = output_dir / f"result_{sample_idx:04d}.npz"
                np.savez(save_path, **save_dict)

    # Compute aggregated metrics
    agg_metrics = {
        "num_samples": len(overall_metrics),
    }

    # Per-channel aggregates
    for wl in wavelengths:
        agg_metrics[f"mae_{wl}"] = np.mean([m["mae"] for m in all_metrics[wl]])
        agg_metrics[f"correlation_{wl}"] = np.mean([m["correlation"] for m in all_metrics[wl]])

    # Overall aggregates
    agg_metrics["mae_overall"] = np.mean([m["mae_overall"] for m in overall_metrics])
    agg_metrics["correlation_overall"] = np.mean([m["corr_overall"] for m in overall_metrics])

    # Save per-sample metrics to CSV
    csv_path = output_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=overall_metrics[0].keys())
        writer.writeheader()
        writer.writerows(overall_metrics)

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

    # Create test dataset
    test_dataset = TestDataset(
        data_dir=f"{config.data_dir}/test",
        wavelengths=wavelengths,
        mag_range=config.mag_range,
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
        wavelengths,
        mag_range=config.mag_range,
    )

    # Print results
    print("Test Results:")
    print(f"  Overall MAE:         {metrics['mae_overall']:.6f}")
    print(f"  Overall Correlation: {metrics['correlation_overall']:.6f}")
    print()
    print("  Per-channel metrics:")
    for wl in wavelengths:
        print(f"    {wl}A: MAE={metrics[f'mae_{wl}']:.6f}, CC={metrics[f'correlation_{wl}']:.4f}")
    print(f"  Samples: {metrics['num_samples']}")
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
        f.write(f"Wavelengths: {wavelengths}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Overall MAE:         {metrics['mae_overall']:.6f}\n")
        f.write(f"Overall Correlation: {metrics['correlation_overall']:.6f}\n")
        f.write("-" * 40 + "\n")
        f.write("Per-channel metrics:\n")
        for wl in wavelengths:
            f.write(f"  {wl}A: MAE={metrics[f'mae_{wl}']:.6f}, CC={metrics[f'correlation_{wl}']:.4f}\n")


if __name__ == "__main__":
    main()
