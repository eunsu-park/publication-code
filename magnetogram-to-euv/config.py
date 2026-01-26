"""
Configuration for Magnetogram to UV/EUV Image Translation.

Park et al. (2019), ApJL, 884, L23
https://doi.org/10.3847/2041-8213/ab46bb
"""

from dataclasses import dataclass

from egghouse.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration for Pix2Pix Magnetogram-to-EUV Generator."""

    # Model architecture
    in_channels: int = 1  # Magnetogram
    out_channels: int = 1  # Single EUV/UV passband (train separately for each)
    ngf: int = 64  # Generator base features
    ndf: int = 64  # Discriminator base features

    # Training mode
    use_gan: bool = True  # Model B (L1+cGAN) vs Model A (L1 only)

    # Training parameters
    epochs: int = 200
    batch_size: int = 1
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Loss weights
    lambda_l1: float = 100.0

    # Target passband (one of 9 passbands)
    # Options: 94, 131, 171, 193, 211, 304, 335, 1600, 1700
    target_passband: int = 304

    # Data
    input_size: int = 1024
    data_dir: str = "./data"

    # Output
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 10

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for Magnetogram-to-EUV Generation."""

    # Model
    in_channels: int = 1
    out_channels: int = 1
    ngf: int = 64
    checkpoint_path: str = "./checkpoints/generator.pth"

    # Target passband
    target_passband: int = 304

    # Data
    input_size: int = 1024
    input_path: str = "./data/magnetogram"
    output_path: str = "./data/euv"

    # Device
    device: str = "cuda"


@dataclass
class MultiPassbandConfig(BaseConfig):
    """Configuration for generating all 9 passbands."""

    # Passbands to generate
    passbands: str = "94,131,171,193,211,304,335,1600,1700"

    # Checkpoint directory (expects generator_{passband}.pth files)
    checkpoint_dir: str = "./checkpoints"

    # Data
    input_size: int = 1024
    input_path: str = "./data/magnetogram"
    output_path: str = "./data/euv_all"

    # Device
    device: str = "cuda"


if __name__ == "__main__":
    config = TrainConfig()
    print(config)

    config_cli = TrainConfig.from_args([
        "--target_passband", "171",
        "--use_gan",
        "--lr", "0.0001"
    ])
    print(f"\nFrom CLI: passband={config_cli.target_passband}, use_gan={config_cli.use_gan}")
