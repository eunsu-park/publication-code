"""
Training Script for Solar Flare Forecasting.

Park et al. (2018), ApJ, 869, 91
https://doi.org/10.3847/1538-4357/aaed40

This script trains a CNN model for binary flare prediction.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --model_type proposed --lr 0.001
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from dataset import TrainDataset, ValidationDataset, get_positive_classes
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
    }


class Trainer:
    """
    Trainer for flare prediction CNN.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        config: Training configuration.
        model: CNN model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        class_weights: Optional class weights for loss function.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Model
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        if config.lr_scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.lr_step_size,
                gamma=config.lr_gamma,
            )
        elif config.lr_scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
            )
        else:
            self.scheduler = None

        # Loss function
        if class_weights is not None and config.class_weight:
            class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)

        # Checkpoints
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self.history: List[Dict] = []
        self.global_step = 0
        self.best_val_tss = -float("inf")

    def train_step(
        self, input_data: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Single training step.

        Args:
            input_data: Input magnetogram batch.
            labels: Target labels.

        Returns:
            Tuple of (loss, predictions, labels).
        """
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        logits = self.model(input_data)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Get predictions
        preds = torch.argmax(logits, dim=1)

        return loss.item(), preds, labels

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input_data, labels in self.val_loader:
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(input_data)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * input_data.size(0)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds)
                all_labels.append(labels)

        self.model.train()

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics["val_loss"] = total_loss / len(self.val_loader.dataset)

        return metrics

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_idx, (input_data, labels) in enumerate(self.train_loader):
            loss, preds, batch_labels = self.train_step(input_data, labels)

            total_loss += loss * input_data.size(0)
            all_preds.append(preds)
            all_labels.append(batch_labels)
            self.global_step += 1

            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar("train/loss_step", loss, self.global_step)

        # Concatenate all predictions and labels
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics["train_loss"] = total_loss / len(self.train_loader.dataset)

        return metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_tss": self.best_val_tss,
            "config": {
                "model_type": self.config.model_type,
                "in_channels": self.config.in_channels,
                "num_classes": self.config.num_classes,
                "flare_threshold": self.config.flare_threshold,
                "growth_rate": self.config.growth_rate,
                "num_modules": self.config.num_modules,
                "blocks_per_module": self.config.blocks_per_module,
                "init_features": self.config.init_features,
            },
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pth")

        # Save periodic
        if (epoch + 1) % self.config.save_interval == 0:
            torch.save(checkpoint, self.save_dir / f"checkpoint_epoch_{epoch+1}.pth")

        # Save best
        if is_best:
            torch.save(checkpoint, self.save_dir / "checkpoint_best.pth")
            torch.save(
                self.model.state_dict(), self.save_dir / "model_best.pth"
            )

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.

        Returns:
            Starting epoch number.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_tss = checkpoint["best_val_tss"]

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint["epoch"] + 1

    def save_history(self) -> None:
        """Save training history to CSV."""
        csv_path = self.log_dir / "history.csv"

        if len(self.history) == 0:
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

    def fit(self, epochs: int, start_epoch: int = 0) -> None:
        """
        Train the model.

        Args:
            epochs: Total number of epochs.
            start_epoch: Starting epoch (for resuming).
        """
        print(f"Starting training from epoch {start_epoch + 1}")
        print(f"Model type: {self.config.model_type}")
        print(f"Flare threshold: {self.config.flare_threshold}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 60)

        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.config.lr

            # Log to tensorboard
            self.writer.add_scalar("epoch/train_loss", train_metrics["train_loss"], epoch)
            self.writer.add_scalar("epoch/val_loss", val_metrics["val_loss"], epoch)
            self.writer.add_scalar("epoch/train_accuracy", train_metrics["accuracy"], epoch)
            self.writer.add_scalar("epoch/val_accuracy", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("epoch/val_tss", val_metrics["tss"], epoch)
            self.writer.add_scalar("epoch/val_hss", val_metrics["hss"], epoch)
            self.writer.add_scalar("epoch/lr", current_lr, epoch)

            # Check if best (use TSS as primary metric)
            is_best = val_metrics["tss"] > self.best_val_tss
            if is_best:
                self.best_val_tss = val_metrics["tss"]

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Record history
            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["train_loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["val_loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_tss": val_metrics["tss"],
                "val_hss": val_metrics["hss"],
                "lr": current_lr,
                "is_best": is_best,
            }
            self.history.append(record)

            # Print progress
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Loss: {train_metrics['train_loss']:.4f}/{val_metrics['val_loss']:.4f} "
                f"Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} "
                f"TSS: {val_metrics['tss']:.4f} "
                f"{'*' if is_best else ''}"
            )

        # Save final history
        self.save_history()
        self.writer.close()

        print("-" * 60)
        print(f"Training completed. Best val TSS: {self.best_val_tss:.4f}")


def main() -> None:
    """Main training function."""
    # Load config
    config = TrainConfig.from_args()

    # Save config for reproducibility
    config_save_path = Path(config.save_dir) / "config.yaml"
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_save_path))

    print(f"Model type: {config.model_type}")
    print(f"Flare threshold: {config.flare_threshold}")
    print(f"Positive classes: {get_positive_classes(config.flare_threshold)}")

    # Create datasets
    train_dataset = TrainDataset(
        data_dir=f"{config.data_dir}/train",
        flare_threshold=config.flare_threshold,
        mag_range=config.mag_range,
        random_flip=config.random_flip,
        random_rotation=config.random_rotation,
    )
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        flare_threshold=config.flare_threshold,
        mag_range=config.mag_range,
    )

    print(f"Training set - Negative: {train_dataset.class_counts['negative']}, "
          f"Positive: {train_dataset.class_counts['positive']}")
    print(f"Validation set - Negative: {val_dataset.class_counts['negative']}, "
          f"Positive: {val_dataset.class_counts['positive']}")

    # Get class weights
    class_weights = train_dataset.get_class_weights() if config.class_weight else None
    if class_weights is not None:
        print(f"Class weights: {class_weights.tolist()}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Create model
    model = get_model(
        model_type=config.model_type,
        in_channels=config.in_channels,
        num_classes=config.num_classes,
        growth_rate=config.growth_rate,
        num_modules=config.num_modules,
        blocks_per_module=config.blocks_per_module,
        init_features=config.init_features,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
    )

    # Train
    trainer.fit(epochs=config.epochs)


if __name__ == "__main__":
    main()
