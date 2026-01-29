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

    # Wavelength selection (comma-separated)
    # Available: 94, 131, 171, 193, 211, 335
    input_wavelengths: str = "171,193,211"   # 3 input channels
    output_wavelengths: str = "94,131,335"   # 3 output channels

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Loss function
    loss_type: str = "mse"  # "mse" or "l1"

    # Data
    data_dir: str = "./data"

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

    @property
    def input_wavelength_list(self) -> list:
        """Parse input_wavelengths string to list of integers."""
        return [int(w.strip()) for w in self.input_wavelengths.split(",")]

    @property
    def output_wavelength_list(self) -> list:
        """Parse output_wavelengths string to list of integers."""
        return [int(w.strip()) for w in self.output_wavelengths.split(",")]

    @property
    def in_channels(self) -> int:
        """Number of input channels based on selected wavelengths."""
        return len(self.input_wavelength_list)

    @property
    def out_channels(self) -> int:
        """Number of output channels based on selected wavelengths."""
        return len(self.output_wavelength_list)


@dataclass
class InferenceConfig(BaseConfig):
    """Inference configuration for FCN Pixel Translator."""

    # Model
    model_type: str = "fcn"
    input_wavelengths: str = "171,193,211"
    output_wavelengths: str = "94,131,335"
    checkpoint_path: str = "./checkpoints/checkpoint_best.pth"

    # Data
    data_dir: str = "./data"
    output_dir: str = "./results"

    # Device
    device: str = "cuda"
    batch_size: int = 1

    @property
    def input_wavelength_list(self) -> list:
        return [int(w.strip()) for w in self.input_wavelengths.split(",")]

    @property
    def output_wavelength_list(self) -> list:
        return [int(w.strip()) for w in self.output_wavelengths.split(",")]

    @property
    def in_channels(self) -> int:
        return len(self.input_wavelength_list)

    @property
    def out_channels(self) -> int:
        return len(self.output_wavelength_list)


if __name__ == "__main__":
    config = TrainConfig()
    print(config)
    print(f"Input wavelengths: {config.input_wavelength_list}")
    print(f"Output wavelengths: {config.output_wavelength_list}")
    print(f"in_channels: {config.in_channels}, out_channels: {config.out_channels}")
