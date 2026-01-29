"""
Test Script for Solar Flare Forecasting.

Park et al. (2018), ApJ, 869, 91
https://doi.org/10.3847/1538-4357/aaed40

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
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import InferenceConfig
from dataset import TestDataset
from networks import get_model


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: Predicted labels.
        labels: Ground truth labels.

    Returns:
        Dictionary of metrics.
    """
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    tn = np.sum((preds == 0) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)

    # Precision (Positive predictive value)
    precision = tp / (tp + fp + 1e-8)

    # Recall (True positive rate, sensitivity)
    recall = tp / (tp + fn + 1e-8)

    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # True skill statistic (TSS)
    tss = recall - fp / (fp + tn + 1e-8)

    # Heidke skill score (HSS)
    expected = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn + 1e-8)
    hss = (tp + tn - expected) / (tp + fp + tn + fn - expected + 1e-8)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tss": float(tss),
        "hss": float(hss),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def predict_and_save(
    model: nn.Module,
    test_loader: DataLoader,
    test_dataset: TestDataset,
    output_dir: Path,
    device: torch.device,
    mag_range: float,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Run inference, compute metrics, and save predictions.

    Args:
        model: CNN model.
        test_loader: Test data loader.
        test_dataset: Test dataset (for file paths).
        output_dir: Directory to save results.
        device: Device to run inference on.
        mag_range: Magnetogram normalization range for denormalization.

    Returns:
        Tuple of (overall_metrics, per_sample_metrics).
    """
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    all_labels = []
    all_probs = []
    per_sample_metrics = []

    sample_idx = 0

    with torch.no_grad():
        for batch_idx, (input_data, labels) in enumerate(test_loader):
            input_data_device = input_data.to(device)

            # Forward pass
            logits = model(input_data_device)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Move to numpy
            input_np = input_data.cpu().numpy()
            labels_np = labels.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = probs.cpu().numpy()

            batch_size = input_data.size(0)

            for i in range(batch_size):
                # Get file path
                filepath = test_dataset.get_filepath(sample_idx)
                filename = filepath.stem

                # Extract single sample
                inp = input_np[i, 0]  # Remove channel dimension
                label = labels_np[i]
                pred = preds_np[i]
                prob = probs_np[i]

                # Denormalize input
                inp_denorm = inp * mag_range

                # Record metrics
                sample_metrics = {
                    "filename": filename,
                    "sample_idx": sample_idx,
                    "label": int(label),
                    "prediction": int(pred),
                    "prob_negative": float(prob[0]),
                    "prob_positive": float(prob[1]),
                    "correct": int(pred == label),
                }
                per_sample_metrics.append(sample_metrics)

                all_preds.append(pred)
                all_labels.append(label)
                all_probs.append(prob)

                # Save npz
                np.savez(
                    output_dir / f"{filename}.npz",
                    input=inp_denorm,
                    label=label,
                    prediction=pred,
                    probability=prob,
                )

                sample_idx += 1

    # Compute overall metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_metrics = compute_metrics(all_preds, all_labels)
    overall_metrics["num_samples"] = len(all_preds)

    return overall_metrics, per_sample_metrics


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

    # Get model config from checkpoint or use config
    ckpt_config = checkpoint.get("config", {})
    model_type = ckpt_config.get("model_type", config.model_type)
    in_channels = ckpt_config.get("in_channels", config.in_channels)
    num_classes = ckpt_config.get("num_classes", config.num_classes)
    flare_threshold = ckpt_config.get("flare_threshold", "c")
    growth_rate = ckpt_config.get("growth_rate", config.growth_rate)
    num_modules = ckpt_config.get("num_modules", config.num_modules)
    blocks_per_module = ckpt_config.get("blocks_per_module", config.blocks_per_module)

    # Create model
    model = get_model(
        model_type=model_type,
        in_channels=in_channels,
        num_classes=num_classes,
        growth_rate=growth_rate,
        num_modules=num_modules,
        blocks_per_module=blocks_per_module,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Model type: {model_type}")
    print(f"Flare threshold: {flare_threshold}")

    # Create test dataset
    test_dataset = TestDataset(
        data_dir=f"{config.data_dir}/test",
        flare_threshold=flare_threshold,
        mag_range=config.mag_range,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"Test samples: {len(test_dataset)}")
    print(f"  Negative: {test_dataset.class_counts['negative']}")
    print(f"  Positive: {test_dataset.class_counts['positive']}")
    print(f"Device: {device}")
    print("-" * 50)

    # Run inference and save
    output_dir = Path(config.output_dir)
    overall_metrics, per_sample_metrics = predict_and_save(
        model, test_loader, test_dataset, output_dir, device, config.mag_range
    )

    # Print results
    print("Test Results:")
    print(f"  Accuracy:  {overall_metrics['accuracy']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall:    {overall_metrics['recall']:.4f}")
    print(f"  F1:        {overall_metrics['f1']:.4f}")
    print(f"  TSS:       {overall_metrics['tss']:.4f}")
    print(f"  HSS:       {overall_metrics['hss']:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}")
    print(f"    FN: {overall_metrics['fn']}, TN: {overall_metrics['tn']}")
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
        f.write(f"Model type: {model_type}\n")
        f.write(f"Flare threshold: {flare_threshold}\n")
        f.write(f"Samples: {overall_metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:  {overall_metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {overall_metrics['precision']:.4f}\n")
        f.write(f"Recall:    {overall_metrics['recall']:.4f}\n")
        f.write(f"F1:        {overall_metrics['f1']:.4f}\n")
        f.write(f"TSS:       {overall_metrics['tss']:.4f}\n")
        f.write(f"HSS:       {overall_metrics['hss']:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}\n")
        f.write(f"  FN: {overall_metrics['fn']}, TN: {overall_metrics['tn']}\n")

    print(f"Summary saved to: {summary_path}")
    print(f"Predictions saved to: {output_dir}")


if __name__ == "__main__":
    main()
