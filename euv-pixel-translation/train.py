"""
Training Script for EUV Pixel-to-Pixel Translation.

Park et al. (2023), ApJS, 264, 33
https://doi.org/10.3847/1538-4365/aca902

This script trains an FCN model for pixel-level translation between EUV images.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --lr 0.0001 --epochs 200
"""

import csv
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from dataset import TrainDataset, ValidationDataset
from networks import get_model


class Trainer:
    """
    Trainer for FCN/CNN EUV translation model.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        config: Training configuration.
        model: FCN or CNN model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        config: TrainConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Model
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        # Loss function
        if config.loss_type.lower() == "mse":
            self.criterion = nn.MSELoss()
        elif config.loss_type.lower() == "l1":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {config.loss_type}")

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
        self.best_val_loss = float("inf")

    def train_step(
        self, input_data: torch.Tensor, target_data: torch.Tensor
    ) -> float:
        """
        Single training step.

        Args:
            input_data: Input EUV image batch.
            target_data: Target EUV image batch.

        Returns:
            Loss value.
        """
        input_data = input_data.to(self.device)
        target_data = target_data.to(self.device)

        self.optimizer.zero_grad()

        # Forward pass
        output = self.model(input_data)

        # Compute loss
        loss = self.criterion(output, target_data)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for input_data, target_data in self.val_loader:
                input_data = input_data.to(self.device)
                target_data = target_data.to(self.device)

                output = self.model(input_data)
                loss = self.criterion(output, target_data)

                total_loss += loss.item() * input_data.size(0)
                total_samples += input_data.size(0)

        self.model.train()

        avg_loss = total_loss / total_samples

        return {"val_loss": avg_loss}

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
        num_batches = 0

        for batch_idx, (input_data, target_data) in enumerate(self.train_loader):
            loss = self.train_step(input_data, target_data)

            total_loss += loss
            num_batches += 1
            self.global_step += 1

            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar("train/loss", loss, self.global_step)

        avg_loss = total_loss / num_batches

        return {"train_loss": avg_loss}

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
            "best_val_loss": self.best_val_loss,
            "input_wavelengths": self.config.input_wavelength_list,
            "output_wavelengths": self.config.output_wavelength_list,
        }

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
        self.best_val_loss = checkpoint["best_val_loss"]

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
        print(f"Input wavelengths: {self.config.input_wavelength_list}")
        print(f"Output wavelengths: {self.config.output_wavelength_list}")
        print(f"Loss type: {self.config.loss_type}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 50)

        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Log to tensorboard
            self.writer.add_scalar(
                "epoch/train_loss", train_metrics["train_loss"], epoch
            )
            self.writer.add_scalar(
                "epoch/val_loss", val_metrics["val_loss"], epoch
            )

            # Check if best
            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Record history
            record = {
                "epoch": epoch + 1,
                "train_loss": train_metrics["train_loss"],
                "val_loss": val_metrics["val_loss"],
                "is_best": is_best,
            }
            self.history.append(record)

            # Print progress
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train: {train_metrics['train_loss']:.6f} "
                f"Val: {val_metrics['val_loss']:.6f} "
                f"{'*' if is_best else ''}"
            )

        # Save final history
        self.save_history()
        self.writer.close()

        print("-" * 50)
        print(f"Training completed. Best val loss: {self.best_val_loss:.6f}")


def main() -> None:
    """Main training function."""
    # Load config
    config = TrainConfig.from_args()

    # Save config for reproducibility
    config_save_path = Path(config.save_dir) / "config.yaml"
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_save_path))

    print(f"Input wavelengths: {config.input_wavelength_list}")
    print(f"Output wavelengths: {config.output_wavelength_list}")
    print(f"in_channels: {config.in_channels}, out_channels: {config.out_channels}")

    # Create datasets
    train_dataset = TrainDataset(
        data_dir=f"{config.data_dir}/train",
        input_wavelengths=config.input_wavelength_list,
        output_wavelengths=config.output_wavelength_list,
        model_type=config.model_type,
    )
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        input_wavelengths=config.input_wavelength_list,
        output_wavelengths=config.output_wavelength_list,
        model_type=config.model_type,
    )

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
        out_channels=config.out_channels,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Train
    trainer.fit(epochs=config.epochs)


if __name__ == "__main__":
    main()
