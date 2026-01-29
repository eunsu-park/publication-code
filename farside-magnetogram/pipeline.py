"""
Training Pipeline for Far-side Magnetogram Generation.

Kim, Park et al. (2019), Nature Astronomy, 3, 397
https://doi.org/10.1038/s41550-019-0711-5

This pipeline trains a Pix2Pix model to generate magnetograms from EUV images.
- Train/Valid Input: SDO/AIA 304 nm
- Train/Valid Target: SDO/HMI magnetogram
- Test Input: STEREO/EUVI 304 nm (far-side)
"""

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import TrainConfig
from networks import Generator, Discriminator


def normalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Normalize EUV 304 nm image.

    Args:
        data: Raw EUV data.

    Returns:
        Normalized data in approximately [-1, 1] range.
    """
    data = np.clip(data + 1, 1, None)
    data = np.log2(data)
    data = (data / 7) - 1.0
    return data


def normalize_magnetogram(data: np.ndarray, data_range: float = 100.0) -> np.ndarray:
    """
    Normalize magnetogram.

    Args:
        data: Raw magnetogram data.
        data_range: Normalization factor.

    Returns:
        Normalized data.
    """
    return data / data_range


def denormalize_magnetogram(data: np.ndarray, data_range: float = 100.0) -> np.ndarray:
    """
    Denormalize magnetogram.

    Args:
        data: Normalized magnetogram data.
        data_range: Normalization factor.

    Returns:
        Denormalized data.
    """
    return data * data_range


class TrainDataset(Dataset):
    """
    Training dataset for far-side magnetogram generation.

    Loads .npz files containing AIA 304 nm and HMI magnetogram pairs.
    - Input: aia_304 (1024×1024)
    - Target: hmi_mag (1024×1024)

    Args:
        data_dir: Directory containing .npz files.
        data_range: Magnetogram normalization factor (default: 100.0).
    """

    def __init__(
        self,
        data_dir: str,
        data_range: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npz file
        data = np.load(self.file_list[idx])

        # Input: AIA 304 nm
        aia_304 = data["aia_304"].astype(np.float32)
        aia_304 = normalize_euv(aia_304)

        # Target: HMI magnetogram
        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.data_range)

        # Add channel dimension: (H, W) -> (1, H, W)
        if aia_304.ndim == 2:
            aia_304 = aia_304[np.newaxis, ...]
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        return (
            torch.from_numpy(aia_304).float(),
            torch.from_numpy(hmi_mag).float(),
        )


class ValidationDataset(Dataset):
    """
    Validation dataset for far-side magnetogram generation.

    Same format as TrainDataset.
    """

    def __init__(
        self,
        data_dir: str,
        data_range: float = 100.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_range = data_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.file_list[idx])

        aia_304 = data["aia_304"].astype(np.float32)
        aia_304 = normalize_euv(aia_304)

        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.data_range)

        if aia_304.ndim == 2:
            aia_304 = aia_304[np.newaxis, ...]
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        return (
            torch.from_numpy(aia_304).float(),
            torch.from_numpy(hmi_mag).float(),
        )


class TestDataset(Dataset):
    """
    Test dataset for far-side magnetogram generation.

    Loads .npz files containing STEREO/EUVI 304 nm images.
    - Input: euvi_304 (1024×1024)
    - No target (inference only)

    Args:
        data_dir: Directory containing .npz files.
    """

    def __init__(self, data_dir: str) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        filepath = self.file_list[idx]
        data = np.load(filepath)

        # Input: EUVI 304 nm (same normalization as AIA 304)
        euvi_304 = data["euvi_304"].astype(np.float32)
        euvi_304 = normalize_euv(euvi_304)

        if euvi_304.ndim == 2:
            euvi_304 = euvi_304[np.newaxis, ...]

        return torch.from_numpy(euvi_304).float(), filepath.stem


class Trainer:
    """
    Trainer for Pix2Pix far-side magnetogram generation model.

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
        self, euv: torch.Tensor, magnetogram: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            euv: EUV 304 nm image batch.
            magnetogram: Target magnetogram batch.

        Returns:
            Tuple of (generator_loss, discriminator_loss).
        """
        euv = euv.to(self.device)
        magnetogram = magnetogram.to(self.device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.optimizer_d.zero_grad()

        # Generate fake magnetogram
        fake = self.generator(euv)

        # Real pair
        real_pair = torch.cat([euv, magnetogram], dim=1)
        pred_real = self.discriminator(real_pair)
        real_label = torch.ones_like(pred_real)
        loss_d_real = self.criterion_gan(pred_real, real_label)

        # Fake pair
        fake_pair = torch.cat([euv, fake.detach()], dim=1)
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
        fake_pair = torch.cat([euv, fake], dim=1)
        pred_fake = self.discriminator(fake_pair)
        loss_g_gan = self.criterion_gan(pred_fake, real_label)

        # L1 loss
        loss_g_l1 = self.criterion_l1(fake, magnetogram)

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
            for euv, magnetogram in self.val_loader:
                euv = euv.to(self.device)
                magnetogram = magnetogram.to(self.device)

                fake = self.generator(euv)
                loss_l1 = self.criterion_l1(fake, magnetogram)

                total_l1 += loss_l1.item() * euv.size(0)
                total_samples += euv.size(0)

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

        for batch_idx, (euv, magnetogram) in enumerate(self.train_loader):
            loss_g, loss_d = self.train_step(euv, magnetogram)

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
            self.writer.add_scalar(
                "epoch/val_l1_loss", val_metrics["val_l1_loss"], epoch
            )

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
        data_range=config.data_range,
    )
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
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

    # Train
    trainer.fit(epochs=config.epochs)


if __name__ == "__main__":
    main()
