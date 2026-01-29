"""
Test Script for Far-side Magnetogram Generation.

Kim, Park et al. (2019), Nature Astronomy, 3, 397
https://doi.org/10.1038/s41550-019-0711-5

This script loads a trained checkpoint, evaluates on the test set,
and saves predictions in npz format.

Supports both:
- Near-side test: AIA 304 nm with HMI magnetogram target (evaluation + save)
- Far-side inference: EUVI 304 nm without target (save only)

Usage:
    # Near-side test with evaluation
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results

    # Far-side inference (no target)
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results --farside
"""

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import InferenceConfig
from dataset import TestDataset, denormalize_magnetogram, denormalize_euv
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
    data_range: float = 100.0,
    farside: bool = False,
) -> Dict[str, float]:
    """
    Run inference on test set and save predictions.

    Args:
        model: Generator model.
        test_loader: Test data loader.
        output_dir: Directory to save results.
        device: Device to run inference on.
        data_range: Normalization factor for denormalization.
        farside: If True, skip metric computation (no target).

    Returns:
        Dictionary of aggregated metrics (empty if farside=True).
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    with torch.no_grad():
        for idx, (euv, target) in enumerate(test_loader):
            euv = euv.to(device)

            # Generate output
            output = model(euv)

            # Move to CPU and numpy
            euv_np = euv.cpu().numpy()
            target_np = target.numpy()
            output_np = output.cpu().numpy()

            # Process each sample in batch
            for b in range(euv_np.shape[0]):
                sample_idx = idx * test_loader.batch_size + b

                # Get single sample
                input_sample = euv_np[b]      # (1, H, W)
                target_sample = target_np[b]  # (1, H, W)
                output_sample = output_np[b]  # (1, H, W)

                # Denormalize
                input_denorm = denormalize_euv(input_sample)
                output_denorm = denormalize_magnetogram(output_sample, data_range)

                if not farside:
                    target_denorm = denormalize_magnetogram(target_sample, data_range)

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
                else:
                    # Far-side: no target
                    save_path = output_dir / f"result_{sample_idx:04d}.npz"
                    np.savez(
                        save_path,
                        input=input_denorm,
                        prediction=output_denorm,
                    )

    if farside or len(all_metrics) == 0:
        return {"num_samples": idx * test_loader.batch_size + euv_np.shape[0]}

    # Compute aggregated metrics
    agg_metrics = {
        "mae": np.mean([m["mae"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "rmse": np.mean([m["rmse"] for m in all_metrics]),
        "correlation": np.mean([m["correlation"] for m in all_metrics]),
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
    farside = getattr(config, "farside", False)
    test_dataset = TestDataset(
        data_dir=f"{config.data_dir}/test",
        data_range=config.data_range,
        farside=farside,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Use single worker for ordered saving
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Far-side mode: {farside}")
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
        farside=farside,
    )

    # Print results
    if farside:
        print("Far-side Inference Results:")
        print(f"  Samples: {metrics['num_samples']}")
    else:
        print("Test Results:")
        print(f"  MAE:         {metrics['mae']:.6f}")
        print(f"  MSE:         {metrics['mse']:.6f}")
        print(f"  RMSE:        {metrics['rmse']:.6f}")
        print(f"  Correlation: {metrics['correlation']:.6f}")
        print(f"  Samples:     {metrics['num_samples']}")

    print("-" * 50)
    print(f"Results saved to: {output_dir}")

    # Save summary
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("Test Results Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Far-side mode: {farside}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        if not farside:
            f.write("-" * 40 + "\n")
            f.write(f"MAE:         {metrics['mae']:.6f}\n")
            f.write(f"MSE:         {metrics['mse']:.6f}\n")
            f.write(f"RMSE:        {metrics['rmse']:.6f}\n")
            f.write(f"Correlation: {metrics['correlation']:.6f}\n")


if __name__ == "__main__":
    main()
