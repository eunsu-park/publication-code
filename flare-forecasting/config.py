"""
Configuration for Solar Flare Forecasting.

Park et al. (2018), ApJ, 869, 91
https://doi.org/10.3847/1538-4357/aaed40
"""

from dataclasses import dataclass

from egghouse.config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration for Flare Prediction CNN."""

    # Model architecture
    model_type: str = "proposed"  # "alexnet", "googlenet", "proposed"
    in_channels: int = 1  # Magnetogram
    num_classes: int = 2  # Binary: Flare / No-flare

    # Proposed model specific
    growth_rate: int = 16
    num_modules: int = 4
    blocks_per_module: int = 6
    init_features: int = 16

    # Training parameters
    epochs: int = 100
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9

    # Learning rate scheduler
    lr_scheduler: str = "step"  # "step", "cosine", "none"
    lr_step_size: int = 30
    lr_gamma: float = 0.1

    # Loss and class balancing
    loss_type: str = "cross_entropy"
    class_weight: bool = True  # Use class weights for imbalanced data

    # Data
    input_size: int = 1024
    data_dir: str = "./data"
    flare_threshold: str = "C1.0"  # Minimum flare class

    # Data augmentation
    augmentation: bool = True
    random_flip: bool = True
    random_rotation: bool = True

    # Output
    save_dir: str = "./checkpoints"
    log_interval: int = 50
    save_interval: int = 10

    # Device
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for Flare Prediction."""

    # Model
    model_type: str = "proposed"
    in_channels: int = 1
    num_classes: int = 2
    growth_rate: int = 16
    num_modules: int = 4
    blocks_per_module: int = 6
    checkpoint_path: str = "./checkpoints/best_model.pth"

    # Data
    input_size: int = 1024
    input_path: str = "./data/test"

    # Output
    output_path: str = "./results"
    threshold: float = 0.5  # Probability threshold for positive class

    # Device
    device: str = "cuda"


if __name__ == "__main__":
    config = TrainConfig()
    print(config)

    config_cli = TrainConfig.from_args([
        "--model_type", "proposed",
        "--lr", "0.0001",
        "--epochs", "200"
    ])
    print(f"\nFrom CLI: model={config_cli.model_type}, lr={config_cli.lr}")
