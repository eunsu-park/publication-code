"""
Configuration for EUV Pixel-to-Pixel Translation.

Park et al. (2023), ApJS, 264, 33
https://doi.org/10.3847/1538-4365/aca902
"""

from dataclasses import dataclass

from egghouse.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration for FCN Pixel Translator."""

    # Model architecture
    model_type: str = "fcn"  # "fcn" or "cnn"
    in_channels: int = 3  # AIA 17.1, 19.3, 21.1 nm
    out_channels: int = 3  # AIA 9.4, 13.1, 33.5 nm

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Loss function
    loss_type: str = "mse"  # "mse" or "l1"

    # Data
    data_dir: str = "./data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1

    # Output
    save_dir: str = "./checkpoints"
    log_interval: int = 100
    save_interval: int = 10

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for FCN Pixel Translator."""

    # Model
    model_type: str = "fcn"
    in_channels: int = 3
    out_channels: int = 3
    checkpoint_path: str = "./checkpoints/best_model.pth"

    # Data
    input_path: str = "./data/input"
    output_path: str = "./data/output"

    # Device
    device: str = "cuda"
    batch_size: int = 1


if __name__ == "__main__":
    # Test configuration
    config = TrainConfig()
    print(config)

    # Test CLI
    config_cli = TrainConfig.from_args(["--lr", "0.001", "--epochs", "50"])
    print(f"\nFrom CLI: lr={config_cli.lr}, epochs={config_cli.epochs}")
