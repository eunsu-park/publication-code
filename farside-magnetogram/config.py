"""
Configuration for Far-side Magnetogram Generation.

Kim, Park et al. (2019), Nature Astronomy, 3, 397
https://doi.org/10.1038/s41550-019-0711-5
"""

from dataclasses import dataclass

from egghouse.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration for Pix2Pix Far-side Magnetogram Generator."""

    # Model architecture
    in_channels: int = 1  # EUV 304 nm
    out_channels: int = 1  # Magnetogram
    ngf: int = 64  # Generator base features
    ndf: int = 64  # Discriminator base features

    # Training parameters
    epochs: int = 200
    batch_size: int = 1  # Large images require small batch
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Loss weights
    lambda_l1: float = 100.0

    # Data
    input_size: int = 1024
    data_dir: str = "./data"
    data_range: float = 100.0  # Magnetogram range: -100 to +100 G

    # Output
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    log_interval: int = 100
    save_interval: int = 10

    # Checkpoint (for resuming or validation)
    checkpoint_path: str = "./checkpoints/checkpoint_best.pth"

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for Far-side Magnetogram Generation."""

    # Model
    in_channels: int = 1
    out_channels: int = 1
    ngf: int = 64
    checkpoint_path: str = "./checkpoints/checkpoint_best.pth"

    # Data
    input_size: int = 1024
    data_dir: str = "./data"
    input_path: str = "./data/stereo"
    output_path: str = "./results"
    data_range: float = 100.0
    batch_size: int = 1

    # Far-side mode (EUVI 304 nm without target)
    farside: bool = False

    # Device
    device: str = "cuda"


if __name__ == "__main__":
    config = TrainConfig()
    print(config)

    config_cli = TrainConfig.from_args(["--lr", "0.0001", "--lambda_l1", "50.0"])
    print(f"\nFrom CLI: lr={config_cli.lr}, lambda_l1={config_cli.lambda_l1}")
