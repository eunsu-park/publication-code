"""
Validation Script for Solar Flare Forecasting.

Park et al. (2018), ApJ, 869, 91
https://doi.org/10.3847/1538-4357/aaed40

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
from networks import get_model


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        preds: Predicted labels.
        labels: Ground truth labels.

    Returns:
        Dictionary of metrics.
    """
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

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


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run evaluation on validation set.

    Args:
        model: CNN model.
        val_loader: Validation data loader.
        device: Device to run evaluation on.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_data, labels in val_loader:
            input_data = input_data.to(device)
            labels = labels.to(device)

            logits = model(input_data)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_data.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.append(preds)
            all_labels.append(labels)

    # Concatenate all predictions and labels
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Compute metrics
    metrics = compute_metrics(all_preds, all_labels)
    metrics["val_loss"] = total_loss / len(val_loader.dataset)
    metrics["num_samples"] = len(val_loader.dataset)

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

    # Get model config from checkpoint or use config
    ckpt_config = checkpoint.get("config", {})
    model_type = ckpt_config.get("model_type", config.model_type)
    in_channels = ckpt_config.get("in_channels", config.in_channels)
    num_classes = ckpt_config.get("num_classes", config.num_classes)
    flare_threshold = ckpt_config.get("flare_threshold", config.flare_threshold)
    growth_rate = ckpt_config.get("growth_rate", config.growth_rate)
    num_modules = ckpt_config.get("num_modules", config.num_modules)
    blocks_per_module = ckpt_config.get("blocks_per_module", config.blocks_per_module)
    init_features = ckpt_config.get("init_features", config.init_features)

    # Create model
    model = get_model(
        model_type=model_type,
        in_channels=in_channels,
        num_classes=num_classes,
        growth_rate=growth_rate,
        num_modules=num_modules,
        blocks_per_module=blocks_per_module,
        init_features=init_features,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
    print(f"Model type: {model_type}")
    print(f"Flare threshold: {flare_threshold}")

    # Create validation dataset
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        flare_threshold=flare_threshold,
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
    print(f"  Negative: {val_dataset.class_counts['negative']}")
    print(f"  Positive: {val_dataset.class_counts['positive']}")
    print(f"Device: {device}")
    print("-" * 50)

    # Run evaluation
    metrics = evaluate(model, val_loader, device)

    # Print results
    print("Validation Results:")
    print(f"  Loss:      {metrics['val_loss']:.6f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  TSS:       {metrics['tss']:.4f}")
    print(f"  HSS:       {metrics['hss']:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    TP: {metrics['tp']}, FP: {metrics['fp']}")
    print(f"    FN: {metrics['fn']}, TN: {metrics['tn']}")
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
        f.write(f"Model type: {model_type}\n")
        f.write(f"Flare threshold: {flare_threshold}\n")
        f.write(f"Samples: {metrics['num_samples']}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Loss:      {metrics['val_loss']:.6f}\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1:        {metrics['f1']:.4f}\n")
        f.write(f"TSS:       {metrics['tss']:.4f}\n")
        f.write(f"HSS:       {metrics['hss']:.4f}\n")
        f.write("-" * 40 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(f"  TP: {metrics['tp']}, FP: {metrics['fp']}\n")
        f.write(f"  FN: {metrics['fn']}, TN: {metrics['tn']}\n")

    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
