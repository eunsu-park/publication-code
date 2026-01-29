"""
Training Script for Magnetogram Denoising.

Park et al. (2020), ApJL, 891, L4
https://doi.org/10.3847/2041-8213/ab74d2

This script trains a Pix2Pix model to denoise SDO/HMI magnetograms.
- Input: Single noisy magnetogram (center frame)
- Target: 21-frame stacked (averaged) magnetogram

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --epochs 100 --lr 0.0001
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from dataset import TrainDataset, ValidationDataset
from networks import Generator, Discriminator


class Trainer:
    """
    Trainer for Pix2Pix magnetogram denoising model.

    Handles training loop, validation, checkpointing, and logging.

    Args:
        config: Training configuration.
        generator: Generator model.
        discriminator: Discriminator model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
    """

    def __init__(
        self,
        config: TrainConfig,
        generator: Generator,
        discriminator: Discriminator,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Models
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)

        # Optimizers
        self.optimizer_g = Adam(
            self.generator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )
        self.optimizer_d = Adam(
            self.discriminator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
        )

        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.lambda_l1 = config.lambda_l1

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
        self, noisy: torch.Tensor, target: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            noisy: Noisy magnetogram batch.
            target: Target (stacked) magnetogram batch.

        Returns:
            Tuple of (generator_loss, discriminator_loss).
        """
        noisy = noisy.to(self.device)
        target = target.to(self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()

        # Generate fake image
        fake = self.generator(noisy)

        # Real pair
        real_pair = torch.cat([noisy, target], dim=1)
        pred_real = self.discriminator(real_pair)
        real_label = torch.ones_like(pred_real)
        loss_d_real = self.criterion_gan(pred_real, real_label)

        # Fake pair
        fake_pair = torch.cat([noisy, fake.detach()], dim=1)
        pred_fake = self.discriminator(fake_pair)
        fake_label = torch.zeros_like(pred_fake)
        loss_d_fake = self.criterion_gan(pred_fake, fake_label)

        # Total discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()

        # -----------------
        # Train Generator
        # -----------------
        self.optimizer_g.zero_grad()

        # GAN loss
        fake_pair = torch.cat([noisy, fake], dim=1)
        pred_fake = self.discriminator(fake_pair)
        loss_g_gan = self.criterion_gan(pred_fake, real_label)

        # L1 loss
        loss_g_l1 = self.criterion_l1(fake, target)

        # Total generator loss
        loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item()

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.generator.eval()

        total_l1 = 0.0
        total_samples = 0

        with torch.no_grad():
            for noisy, target in self.val_loader:
                noisy = noisy.to(self.device)
                target = target.to(self.device)

                fake = self.generator(noisy)
                loss_l1 = self.criterion_l1(fake, target)

                total_l1 += loss_l1.item() * noisy.size(0)
                total_samples += noisy.size(0)

        self.generator.train()

        avg_l1 = total_l1 / total_samples

        return {"val_l1_loss": avg_l1}

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary of training metrics.
        """
        self.generator.train()
        self.discriminator.train()

        total_loss_g = 0.0
        total_loss_d = 0.0
        num_batches = 0

        for batch_idx, (noisy, target) in enumerate(self.train_loader):
            loss_g, loss_d = self.train_step(noisy, target)

            total_loss_g += loss_g
            total_loss_d += loss_d
            num_batches += 1
            self.global_step += 1

            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar("train/loss_g", loss_g, self.global_step)
                self.writer.add_scalar("train/loss_d", loss_d, self.global_step)

        avg_loss_g = total_loss_g / num_batches
        avg_loss_d = total_loss_d / num_batches

        return {"train_loss_g": avg_loss_g, "train_loss_d": avg_loss_d}

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
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "best_val_loss": self.best_val_loss,
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
                self.generator.state_dict(), self.save_dir / "generator_best.pth"
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

        self.generator.load_state_dict(checkpoint["generator_state_dict"])
        self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
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
                "epoch/train_loss_g", train_metrics["train_loss_g"], epoch
            )
            self.writer.add_scalar(
                "epoch/train_loss_d", train_metrics["train_loss_d"], epoch
            )
            self.writer.add_scalar("epoch/val_l1_loss", val_metrics["val_l1_loss"], epoch)

            # Check if best
            is_best = val_metrics["val_l1_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_l1_loss"]

            # Save checkpoint
            self.save_checkpoint(epoch, is_best)

            # Record history
            record = {
                "epoch": epoch + 1,
                "train_loss_g": train_metrics["train_loss_g"],
                "train_loss_d": train_metrics["train_loss_d"],
                "val_l1_loss": val_metrics["val_l1_loss"],
                "is_best": is_best,
            }
            self.history.append(record)

            # Print progress
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"G: {train_metrics['train_loss_g']:.4f} "
                f"D: {train_metrics['train_loss_d']:.4f} "
                f"Val L1: {val_metrics['val_l1_loss']:.4f} "
                f"{'*' if is_best else ''}"
            )

        # Save final history
        self.save_history()
        self.writer.close()

        print("-" * 50)
        print(f"Training completed. Best val L1: {self.best_val_loss:.4f}")


def main() -> None:
    """Main training function."""
    # Load config
    config = TrainConfig.from_args()

    # Save config for reproducibility
    config_save_path = Path(config.save_dir) / "config.yaml"
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    config.to_yaml(str(config_save_path))

    # Create datasets
    train_dataset = TrainDataset(
        data_dir=f"{config.data_dir}/train",
        input_size=config.input_size,
        data_range=config.data_range,
    )
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        input_size=config.input_size,
        data_range=config.data_range,
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

    # Create models
    generator = Generator(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_features=config.ngf,
    )
    discriminator = Discriminator(
        in_channels=config.in_channels + config.out_channels,
        base_features=config.ndf,
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # Calculate epochs from iterations
    steps_per_epoch = len(train_loader)
    epochs = config.iterations // steps_per_epoch

    # Train
    trainer.fit(epochs=epochs)


if __name__ == "__main__":
    main()
