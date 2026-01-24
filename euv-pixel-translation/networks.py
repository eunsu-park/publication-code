"""
Fully Connected Network for Pixel-to-Pixel Translation of Solar EUV Images.

This module implements the FCN model from:
Park et al. (2023), "Pixel-to-pixel Translation of Solar Extreme-ultraviolet
Images for DEMs by Fully Connected Networks", ApJS, 264, 33.
https://doi.org/10.3847/1538-4365/aca902

The model translates SDO/AIA 3-channel EUV images (17.1, 19.3, 21.1 nm)
to other 3-channel EUV images (9.4, 13.1, 33.5 nm) on a per-pixel basis.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class FCNPixelTranslator(nn.Module):
    """
    Fully Connected Network for pixel-to-pixel EUV image translation.

    This model uses a U-Net style encoder-decoder architecture with skip
    connections, but replaces all convolution layers with fully connected
    layers. Each pixel is processed independently, which aligns with the
    physical assumption that DEM at each location depends only on local
    plasma conditions.

    Architecture:
        Encoder: 8 FC layers (3 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512)
        Decoder: 8 FC layers with skip connections via concatenation
        Activation: SiLU (Swish) between all layers except final output

    Args:
        in_channels: Number of input channels (default: 3 for AIA 17.1, 19.3, 21.1 nm).
        out_channels: Number of output channels (default: 3 for AIA 9.4, 13.1, 33.5 nm).

    Example:
        >>> model = FCNPixelTranslator()
        >>> # Input shape: (batch, num_pixels, 3) or (batch, H, W, 3)
        >>> x = torch.randn(1, 1000, 3)
        >>> y = model(x)
        >>> print(y.shape)  # (1, 1000, 3)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder layer dimensions: 3 -> 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
        enc_dims = [in_channels, 64, 128, 256, 512, 512, 512, 512, 512]

        # Build encoder layers
        self.encoder = nn.ModuleList()
        for i in range(8):
            self.encoder.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))

        # Decoder layer dimensions with skip connections
        # Skip connections concatenate encoder outputs to decoder inputs
        # dec1: 512 + 512 (enc7) -> 512
        # dec2: 512 + 512 (enc6) -> 512
        # dec3: 512 + 512 (enc5) -> 512
        # dec4: 512 + 512 (enc4) -> 256
        # dec5: 256 + 256 (enc3) -> 128
        # dec6: 128 + 128 (enc2) -> 64
        # dec7: 64 + 64 (enc1) -> 64
        # dec8: 64 -> 3 (no skip, no activation)
        self.decoder = nn.ModuleList([
            nn.Linear(512 + 512, 512),  # dec1: concat with enc7
            nn.Linear(512 + 512, 512),  # dec2: concat with enc6
            nn.Linear(512 + 512, 512),  # dec3: concat with enc5
            nn.Linear(512 + 512, 256),  # dec4: concat with enc4
            nn.Linear(256 + 256, 128),  # dec5: concat with enc3
            nn.Linear(128 + 128, 64),   # dec6: concat with enc2
            nn.Linear(64 + 64, 64),     # dec7: concat with enc1
            nn.Linear(64, out_channels), # dec8: final output, no skip
        ])

        # SiLU (Swish) activation function
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for pixel-to-pixel translation.

        Args:
            x: Input tensor of shape (batch, num_pixels, in_channels) or
               (batch, height, width, in_channels). The last dimension
               should contain the input channel values.

        Returns:
            Output tensor of the same shape as input but with out_channels
            in the last dimension.
        """
        # Store original shape for reshaping output
        original_shape = x.shape
        original_ndim = x.ndim

        # Flatten spatial dimensions if input is 4D (batch, H, W, C)
        if original_ndim == 4:
            batch_size, height, width, channels = original_shape
            x = x.view(batch_size * height * width, channels)
        elif original_ndim == 3:
            batch_size, num_pixels, channels = original_shape
            x = x.view(batch_size * num_pixels, channels)
        elif original_ndim == 2:
            # Already (N, C) format
            pass
        else:
            raise ValueError(
                f"Expected 2D, 3D, or 4D input, got {original_ndim}D"
            )

        # Encoder forward pass - store outputs for skip connections
        enc_outputs = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1:
                x = self.activation(x)
            enc_outputs.append(x)

        # Decoder forward pass with skip connections
        # enc_outputs indices: [enc1, enc2, enc3, enc4, enc5, enc6, enc7, enc8]
        # Skip connections: dec1 uses enc7, dec2 uses enc6, ..., dec7 uses enc1
        for i, layer in enumerate(self.decoder[:-1]):
            # Get skip connection from encoder
            # i=0 -> enc7 (index 6), i=1 -> enc6 (index 5), ...
            skip_idx = 6 - i
            if skip_idx >= 0:
                skip = enc_outputs[skip_idx]
                x = torch.cat([x, skip], dim=-1)
            x = layer(x)
            x = self.activation(x)

        # Final decoder layer (no skip connection, no activation)
        x = self.decoder[-1](x)

        # Reshape output to match input shape
        if original_ndim == 4:
            x = x.view(batch_size, height, width, self.out_channels)
        elif original_ndim == 3:
            x = x.view(batch_size, num_pixels, self.out_channels)

        return x


class CNNPixelTranslator(nn.Module):
    """
    CNN-based model for comparison with FCN model.

    This model has the same overall structure as FCNPixelTranslator but uses
    convolution layers instead of fully connected layers. It includes batch
    normalization layers as mentioned in the paper.

    The main difference from FCN is that CNN uses neighboring pixel information
    to determine output values, while FCN treats each pixel independently.

    Args:
        in_channels: Number of input channels (default: 3).
        out_channels: Number of output channels (default: 3).
        kernel_size: Convolution kernel size (default: 3).

    Note:
        This model is provided for comparison purposes. The paper shows that
        CNN models can produce boundary effects when processing large images
        in patches, while FCN models do not have this issue.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = kernel_size // 2

        # Encoder dimensions
        enc_dims = [in_channels, 64, 128, 256, 512, 512, 512, 512, 512]

        # Build encoder with Conv2d + BatchNorm
        self.encoder = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        for i in range(8):
            self.encoder.append(
                nn.Conv2d(enc_dims[i], enc_dims[i + 1], kernel_size, padding=padding)
            )
            self.encoder_bn.append(nn.BatchNorm2d(enc_dims[i + 1]))

        # Decoder with skip connections
        self.decoder = nn.ModuleList([
            nn.Conv2d(512 + 512, 512, kernel_size, padding=padding),
            nn.Conv2d(512 + 512, 512, kernel_size, padding=padding),
            nn.Conv2d(512 + 512, 512, kernel_size, padding=padding),
            nn.Conv2d(512 + 512, 256, kernel_size, padding=padding),
            nn.Conv2d(256 + 256, 128, kernel_size, padding=padding),
            nn.Conv2d(128 + 128, 64, kernel_size, padding=padding),
            nn.Conv2d(64 + 64, 64, kernel_size, padding=padding),
            nn.Conv2d(64, out_channels, kernel_size, padding=padding),
        ])

        self.decoder_bn = nn.ModuleList([
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(512),
            nn.BatchNorm2d(256),
            nn.BatchNorm2d(128),
            nn.BatchNorm2d(64),
            nn.BatchNorm2d(64),
        ])

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN-based translation.

        Args:
            x: Input tensor of shape (batch, channels, height, width).
               Note: Unlike FCN, CNN expects channel-first format.

        Returns:
            Output tensor of shape (batch, out_channels, height, width).
        """
        # Encoder forward pass
        enc_outputs = []
        for i, (layer, bn) in enumerate(zip(self.encoder, self.encoder_bn)):
            x = layer(x)
            x = bn(x)
            if i < len(self.encoder) - 1:
                x = self.activation(x)
            enc_outputs.append(x)

        # Decoder forward pass with skip connections
        for i, layer in enumerate(self.decoder[:-1]):
            skip_idx = 6 - i
            if skip_idx >= 0:
                skip = enc_outputs[skip_idx]
                x = torch.cat([x, skip], dim=1)
            x = layer(x)
            x = self.decoder_bn[i](x)
            x = self.activation(x)

        # Final layer (no batch norm, no activation)
        x = self.decoder[-1](x)

        return x


def get_model(
    model_type: str = "fcn",
    in_channels: int = 3,
    out_channels: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get model by type.

    Args:
        model_type: Type of model ("fcn" or "cnn").
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        **kwargs: Additional arguments passed to model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type.lower() == "fcn":
        return FCNPixelTranslator(in_channels, out_channels)
    elif model_type.lower() == "cnn":
        return CNNPixelTranslator(in_channels, out_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test FCN model
    print("Testing FCN model...")
    fcn_model = FCNPixelTranslator()

    # Test with different input shapes
    # Shape: (batch, num_pixels, channels)
    x1 = torch.randn(2, 1000, 3)
    y1 = fcn_model(x1)
    print(f"  Input shape: {x1.shape} -> Output shape: {y1.shape}")

    # Shape: (batch, height, width, channels)
    x2 = torch.randn(2, 64, 64, 3)
    y2 = fcn_model(x2)
    print(f"  Input shape: {x2.shape} -> Output shape: {y2.shape}")

    # Shape: (num_pixels, channels)
    x3 = torch.randn(1000, 3)
    y3 = fcn_model(x3)
    print(f"  Input shape: {x3.shape} -> Output shape: {y3.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in fcn_model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test CNN model
    print("\nTesting CNN model...")
    cnn_model = CNNPixelTranslator()

    # CNN expects (batch, channels, height, width)
    x4 = torch.randn(2, 3, 64, 64)
    y4 = cnn_model(x4)
    print(f"  Input shape: {x4.shape} -> Output shape: {y4.shape}")

    num_params_cnn = sum(p.numel() for p in cnn_model.parameters())
    print(f"  Total parameters: {num_params_cnn:,}")

    print("\nAll tests passed!")
