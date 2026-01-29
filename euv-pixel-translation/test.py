"""
Test Script for EUV Pixel-to-Pixel Translation.

Park et al. (2023), ApJS, 264, 33
https://doi.org/10.3847/1538-4365/aca902

This script loads a trained checkpoint, evaluates on the test set,
and saves predictions in npz format.

Usage:
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth
    python test.py --checkpoint ./checkpoints/checkpoint_best.pth --output_dir ./results
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import InferenceConfig
from dataset import TestDataset, denormalize_euv
from networks import get_model


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        prediction: Predicted EUV image.
        target: Target EUV image.

    Returns:
        Dictionary of metrics.
    """
    # Flatten for correlation
    pred_flat = prediction.flatten()
    target_flat = target.flatten()

    # MAE
    mae = np.mean(np.abs(prediction - target))

    # MSE
    mse = np.mean((prediction - target) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # Correlation coefficient
    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        cc = np.corrcoef(pred_flat, target_flat)[0, 1]
    else:
        cc = 0.0

    # PSNR
    max_val = max(target.max(), 1e-10)
    psnr = 10 * np.log10(max_val ** 2 / (mse + 1e-10))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "cc": cc,
        "psnr": psnr,
    }


def predict_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: TestDataset,
    output_dir: Path,
    device: torch.device,
    input_wavelengths: List[int],
    output_wavelengths: List[int],
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run inference, compute metrics, and save predictions.

    Args:
        model: FCN/CNN model.
        test_loader: Test data loader.
        test_dataset: Test dataset (for file paths).
        output_dir: Directory to save results.
        device: Device to run inference on.
        input_wavelengths: List of input wavelengths.
        output_wavelengths: List of output wavelengths.

    Returns:
        Tuple of (overall_metrics, per_sample_metrics).
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    per_channel_metrics = {wl: {"mae": [], "rmse": [], "cc": []} for wl in output_wavelengths}

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (input_data, target_data) in enumerate(test_loader):
            input_data_device = input_data.to(device)
            output = model(input_data_device)

            # Move to numpy
            input_np = input_data.cpu().numpy()
            target_np = target_data.cpu().numpy()
            output_np = output.cpu().numpy()

            batch_size = input_data.size(0)

            for i in range(batch_size):
                # Get file path
                filepath = test_dataset.get_filepath(sample_idx)
                filename = filepath.stem

                # Extract single sample
                inp = input_np[i]
                tgt = target_np[i]
                pred = output_np[i]

                # Denormalize for metrics and saving
                # Handle both FCN (H, W, C) and CNN (C, H, W) formats
                if inp.ndim == 3 and inp.shape[-1] == len(input_wavelengths):
                    # FCN format: (H, W, C)
                    inp_denorm = np.stack([denormalize_euv(inp[..., c]) for c in range(inp.shape[-1])], axis=-1)
                    tgt_denorm = np.stack([denormalize_euv(tgt[..., c]) for c in range(tgt.shape[-1])], axis=-1)
                    pred_denorm = np.stack([denormalize_euv(pred[..., c]) for c in range(pred.shape[-1])], axis=-1)
                else:
                    # CNN format: (C, H, W)
                    inp_denorm = np.stack([denormalize_euv(inp[c]) for c in range(inp.shape[0])], axis=0)
                    tgt_denorm = np.stack([denormalize_euv(tgt[c]) for c in range(tgt.shape[0])], axis=0)
                    pred_denorm = np.stack([denormalize_euv(pred[c]) for c in range(pred.shape[0])], axis=0)

                # Compute overall metrics
                metrics = compute_metrics(pred_denorm, tgt_denorm)
                metrics["filename"] = filename
                metrics["sample_idx"] = sample_idx

                # Compute per-channel metrics
                for c, wl in enumerate(output_wavelengths):
                    if tgt_denorm.ndim == 3 and tgt_denorm.shape[-1] == len(output_wavelengths):
                        # FCN format
                        ch_metrics = compute_metrics(pred_denorm[..., c], tgt_denorm[..., c])
                    else:
                        # CNN format
                        ch_metrics = compute_metrics(pred_denorm[c], tgt_denorm[c])

                    metrics[f"mae_{wl}"] = ch_metrics["mae"]
                    metrics[f"rmse_{wl}"] = ch_metrics["rmse"]
                    metrics[f"cc_{wl}"] = ch_metrics["cc"]

                    per_channel_metrics[wl]["mae"].append(ch_metrics["mae"])
                    per_channel_metrics[wl]["rmse"].append(ch_metrics["rmse"])
                    per_channel_metrics[wl]["cc"].append(ch_metrics["cc"])

                all_metrics.append(metrics)

                # Save npz with per-channel data
                save_dict = {}

                # Save input channels
                for c, wl in enumerate(input_wavelengths):
                    if inp_denorm.ndim == 3 and inp_denorm.shape[-1] == len(input_wavelengths):
                        save_dict[f"input_{wl}"] = inp_denorm[..., c]
                    else:
                        save_dict[f"input_{wl}"] = inp_denorm[c]

                # Save target and prediction channels
                for c, wl in enumerate(output_wavelengths):
                    if tgt_denorm.ndim == 3 and tgt_denorm.shape[-1] == len(output_wavelengths):
                        save_dict[f"target_{wl}"] = tgt_denorm[..., c]
                        save_dict[f"prediction_{wl}"] = pred_denorm[..., c]
                    else:
                        save_dict[f"target_{wl}"] = tgt_denorm[c]
                        save_dict[f"prediction_{wl}"] = pred_denorm[c]

                np.savez(output_dir / f"{filename}.npz", **save_dict)

                sample_idx += 1

    # Compute overall statistics
    overall_metrics = {
        "mae": np.mean([m["mae"] for m in all_metrics]),
        "mse": np.mean([m["mse"] for m in all_metrics]),
        "rmse": np.mean([m["rmse"] for m in all_metrics]),
        "cc": np.mean([m["cc"] for m in all_metrics]),
        "psnr": np.mean([m["psnr"] for m in all_metrics]),
        "num_samples": len(all_metrics),
    }

    # Add per-channel averages
    for wl in output_wavelengths:
        overall_metrics[f"mae_{wl}"] = np.mean(per_channel_metrics[wl]["mae"])
        overall_metrics[f"rmse_{wl}"] = np.mean(per_channel_metrics[wl]["rmse"])
        overall_metrics[f"cc_{wl}"] = np.mean(per_channel_metrics[wl]["cc"])

    return overall_metrics, all_metrics


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
    input_wavelengths = checkpoint.get("input_wavelengths", config.input_wavelength_list)
    output_wavelengths = checkpoint.get("output_wavelengths", config.output_wavelength_list)

    # Create model
    model = get_model(
        model_type=config.model_type,
        in_channels=len(input_wavelengths),
        out_channels=len(output_wavelengths),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Input wavelengths: {input_wavelengths}")
    print(f"Output wavelengths: {output_wavelengths}")

    # Create test dataset
    test_dataset = TestDataset(
        data_dir=f"{config.data_dir}/test",
        input_wavelengths=input_wavelengths,
        output_wavelengths=output_wavelengths,
        model_type=config.model_type,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")
    print("-" * 50)

    # Run inference and save
    output_dir = Path(config.output_dir)
    overall_metrics, per_sample_metrics = predict_and_save(
        model, test_loader, test_dataset, output_dir, device,
        input_wavelengths, output_wavelengths
    )

    # Print results
    print("Test Results:")
    print(f"  MAE:  {overall_metrics['mae']:.6f}")
    print(f"  MSE:  {overall_metrics['mse']:.6f}")
    print(f"  RMSE: {overall_metrics['rmse']:.6f}")
    print(f"  CC:   {overall_metrics['cc']:.6f}")
    print(f"  PSNR: {overall_metrics['psnr']:.2f}")
    print()
    print("  Per-channel metrics:")
    for wl in output_wavelengths:
        print(f"    {wl}A: MAE={overall_metrics[f'mae_{wl}']:.6f}, "
              f"RMSE={overall_metrics[f'rmse_{wl}']:.6f}, "
              f"CC={overall_metrics[f'cc_{wl}']:.6f}")
    print(f"  Samples: {overall_metrics['num_samples']}")
    print("-" * 50)

    # Save metrics to CSV
    csv_path = output_dir / "test_metrics.csv"
    if len(per_sample_metrics) > 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_sample_metrics[0].keys())
            writer.writeheader()
            writer.writerows(per_sample_metrics)
        print(f"Per-sample metrics saved to: {csv_path}")

    # Save summary
    summary_path = output_dir / "test_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Test Results Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch'] + 1}\n")
        f.write(f"Input wavelengths: {input_wavelengths}\n")
        f.write(f"Output wavelengths: {output_wavelengths}\n")
        f.write(f"Samples: {overall_metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"MAE:  {overall_metrics['mae']:.6f}\n")
        f.write(f"MSE:  {overall_metrics['mse']:.6f}\n")
        f.write(f"RMSE: {overall_metrics['rmse']:.6f}\n")
        f.write(f"CC:   {overall_metrics['cc']:.6f}\n")
        f.write(f"PSNR: {overall_metrics['psnr']:.2f}\n")
        f.write("-" * 40 + "\n")
        f.write("Per-channel metrics:\n")
        for wl in output_wavelengths:
            f.write(f"  {wl}A: MAE={overall_metrics[f'mae_{wl}']:.6f}, "
                    f"RMSE={overall_metrics[f'rmse_{wl}']:.6f}, "
                    f"CC={overall_metrics[f'cc_{wl}']:.6f}\n")

    print(f"Summary saved to: {summary_path}")
    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
