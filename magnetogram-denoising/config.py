"""
Configuration for Magnetogram Denoising.

Park et al. (2020), ApJL, 891, L4
https://doi.org/10.3847/2041-8213/ab74d2
"""

from dataclasses import dataclass

from egghouse.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration for DCGAN Magnetogram Denoiser."""

    # Model architecture
    in_channels: int = 1  # Noisy magnetogram
    out_channels: int = 1  # De-noised magnetogram
    ngf: int = 64  # Generator base features
    ndf: int = 64  # Discriminator base features

    # Training parameters
    iterations: int = 500000  # Total iterations
    batch_size: int = 4
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999

    # Loss weights
    lambda_l1: float = 100.0

    # Data
    input_size: int = 256
    data_dir: str = "./data"
    data_range: float = 100.0  # -100 to +100 G

    # Target: 21-frame stacked magnetogram
    stack_frames: int = 21

    # Output
    save_dir: str = "./checkpoints"
    log_interval: int = 1000
    save_interval: int = 10000

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for Magnetogram Denoising."""

    # Model
    in_channels: int = 1
    out_channels: int = 1
    ngf: int = 64
    checkpoint_path: str = "./checkpoints/generator.pth"

    # Data
    input_size: int = 256
    input_path: str = "./data/noisy"
    output_path: str = "./data/denoised"
    data_range: float = 100.0

    # Device
    device: str = "cuda"


if __name__ == "__main__":
    config = TrainConfig()
    print(config)

    config_cli = TrainConfig.from_args([
        "--iterations", "100000",
        "--lambda_l1", "50.0"
    ])
    print(f"\nFrom CLI: iterations={config_cli.iterations}, lambda_l1={config_cli.lambda_l1}")
