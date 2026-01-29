"""
Training Pipeline for Magnetogram to UV/EUV Image Translation.

Park et al. (2019), ApJL, 884, L23
https://doi.org/10.3847/2041-8213/ab46bb

This pipeline trains a Pix2Pix model to generate EUV/UV images from magnetograms.
- Input: SDO/HMI magnetogram
- Output: SDO/AIA EUV/UV images (1 or more wavelengths)
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


# Available wavelengths
WAVELENGTHS = [94, 131, 171, 193, 211, 304, 335, 1600, 1700]


def normalize_magnetogram(data: np.ndarray, mag_range: float = 1000.0) -> np.ndarray:
    """
    Normalize HMI magnetogram.

    Args:
        data: Raw magnetogram data.
        mag_range: Normalization factor (default: 1000.0).

    Returns:
        Normalized data.
    """
    return data / mag_range


def normalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Normalize AIA EUV/UV image.

    Args:
        data: Raw EUV data.

    Returns:
        Normalized data in approximately [-1, 1] range.
    """
    data = np.clip(data + 1, 1, None)
    data = np.log2(data)
    data = (data / 7) - 1.0
    return data


def denormalize_euv(data: np.ndarray) -> np.ndarray:
    """
    Denormalize AIA EUV/UV image.

    Args:
        data: Normalized EUV data.

    Returns:
        Denormalized data.
    """
    data = (data + 1.0) * 7
    data = np.power(2, data) - 1
    return data


class BaseDataset(Dataset):
    """
    Base dataset for magnetogram-to-EUV translation.

    Loads .npz files containing HMI magnetogram and AIA EUV images.
    - Input: hmi_mag (1024×1024)
    - Output: aia_{wavelength} (1024×1024) × N channels

    Args:
        data_dir: Directory containing .npz files.
        wavelengths: List of target wavelengths.
        mag_range: Magnetogram normalization factor.
    """

    def __init__(
        self,
        data_dir: str,
        wavelengths: List[int],
        mag_range: float = 1000.0,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.wavelengths = wavelengths
        self.mag_range = mag_range
        self.file_list = sorted(self.data_dir.glob("*.npz"))

        if len(self.file_list) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Validate wavelengths
        for wl in wavelengths:
            if wl not in WAVELENGTHS:
                raise ValueError(
                    f"Invalid wavelength {wl}. "
                    f"Available: {WAVELENGTHS}"
                )

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load npz file
        data = np.load(self.file_list[idx])

        # Input: HMI magnetogram
        hmi_mag = data["hmi_mag"].astype(np.float32)
        hmi_mag = normalize_magnetogram(hmi_mag, self.mag_range)

        # Add channel dimension: (H, W) -> (1, H, W)
        if hmi_mag.ndim == 2:
            hmi_mag = hmi_mag[np.newaxis, ...]

        # Output: AIA EUV images (concatenated along channel dimension)
        aia_channels = []
        for wl in self.wavelengths:
            aia = data[f"aia_{wl}"].astype(np.float32)
            aia = normalize_euv(aia)
            if aia.ndim == 2:
                aia = aia[np.newaxis, ...]
            aia_channels.append(aia)

        # Concatenate channels: (N, H, W)
        aia_stack = np.concatenate(aia_channels, axis=0)

        return (
            torch.from_numpy(hmi_mag).float(),
            torch.from_numpy(aia_stack).float(),
        )


class TrainDataset(BaseDataset):
    """Training dataset for magnetogram-to-EUV translation."""

    pass


class ValidationDataset(BaseDataset):
    """Validation dataset for magnetogram-to-EUV translation."""

    pass


class TestDataset(BaseDataset):
    """Test dataset for magnetogram-to-EUV translation."""

    pass


class Trainer:
    """
    Trainer for Pix2Pix magnetogram-to-EUV model.

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
        self.use_gan = config.use_gan

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
        self, magnetogram: torch.Tensor, euv: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            magnetogram: HMI magnetogram batch.
            euv: Target AIA EUV batch.

        Returns:
            Tuple of (generator_loss, discriminator_loss).
        """
        magnetogram = magnetogram.to(self.device)
        euv = euv.to(self.device)

        # Generate fake EUV
        fake = self.generator(magnetogram)

        loss_d = torch.tensor(0.0)

        if self.use_gan:
            # ---------------------
            # Train Discriminator
            # ---------------------
            self.optimizer_d.zero_grad()

            # Real pair
            real_pair = torch.cat([magnetogram, euv], dim=1)
            pred_real = self.discriminator(real_pair)
            real_label = torch.ones_like(pred_real)
            loss_d_real = self.criterion_gan(pred_real, real_label)

            # Fake pair
            fake_pair = torch.cat([magnetogram, fake.detach()], dim=1)
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

        # L1 loss
        loss_g_l1 = self.criterion_l1(fake, euv)

        if self.use_gan:
            # GAN loss
            fake_pair = torch.cat([magnetogram, fake], dim=1)
            pred_fake = self.discriminator(fake_pair)
            loss_g_gan = self.criterion_gan(pred_fake, real_label)
            loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1
        else:
            # L1 only (Model A)
            loss_g = loss_g_l1

        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.item(), loss_d.item() if self.use_gan else 0.0

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
            for magnetogram, euv in self.val_loader:
                magnetogram = magnetogram.to(self.device)
                euv = euv.to(self.device)

                fake = self.generator(magnetogram)
                loss_l1 = self.criterion_l1(fake, euv)

                total_l1 += loss_l1.item() * magnetogram.size(0)
                total_samples += magnetogram.size(0)

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
        if self.use_gan:
            self.discriminator.train()

        total_loss_g = 0.0
        total_loss_d = 0.0
        num_batches = 0

        for batch_idx, (magnetogram, euv) in enumerate(self.train_loader):
            loss_g, loss_d = self.train_step(magnetogram, euv)

            total_loss_g += loss_g
            total_loss_d += loss_d
            num_batches += 1
            self.global_step += 1

            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                self.writer.add_scalar("train/loss_g", loss_g, self.global_step)
                if self.use_gan:
                    self.writer.add_scalar("train/loss_d", loss_d, self.global_step)

        avg_loss_g = total_loss_g / num_batches
        avg_loss_d = total_loss_d / num_batches if self.use_gan else 0.0

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
            "wavelengths": self.config.wavelength_list,
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
        print(f"Target wavelengths: {self.config.wavelength_list}")
        print(f"Output channels: {self.config.out_channels}")
        print(f"Training mode: {'L1+cGAN' if self.use_gan else 'L1 only'}")
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
            if self.use_gan:
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
            d_str = f"D: {train_metrics['train_loss_d']:.4f} " if self.use_gan else ""
            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"G: {train_metrics['train_loss_g']:.4f} "
                f"{d_str}"
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

    print(f"Target wavelengths: {config.wavelength_list}")
    print(f"Output channels: {config.out_channels}")

    # Create datasets
    train_dataset = TrainDataset(
        data_dir=f"{config.data_dir}/train",
        wavelengths=config.wavelength_list,
        mag_range=config.mag_range,
    )
    val_dataset = ValidationDataset(
        data_dir=f"{config.data_dir}/valid",
        wavelengths=config.wavelength_list,
        mag_range=config.mag_range,
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
