"""
DCGAN Networks for Solar Magnetogram Denoising.

This module implements the Generator and Discriminator from:
Park et al. (2020), "De-noising SDO/HMI Solar Magnetograms by Image
Translation Method Based on Deep Learning", ApJL, 891, L4.
https://doi.org/10.3847/2041-8213/ab74d2

The architecture is based on Pix2Pix / DCGAN:
- Generator: U-Net with skip connections
- Discriminator: PatchGAN (70x70 receptive field)

The model translates single noisy magnetograms to de-noised ones
(equivalent to 21-frame stacked magnetograms).
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """
    Encoder block: Conv2d -> BatchNorm -> LeakyReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        use_bn: Whether to use batch normalization.
        negative_slope: Negative slope for LeakyReLU.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_bn: bool = True,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=not use_bn,
            )
        ]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(negative_slope, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """
    Decoder block: ConvTranspose2d -> BatchNorm -> Dropout -> ReLU.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        use_dropout: Whether to use dropout.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        use_dropout: bool = False,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]

        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """
    U-Net Generator for magnetogram denoising.

    Architecture follows Pix2Pix with 8 encoder and 8 decoder layers.
    Skip connections concatenate encoder features to decoder features.

    Args:
        in_channels: Number of input channels (default: 1 for magnetogram).
        out_channels: Number of output channels (default: 1 for de-noised).
        base_features: Base number of features (default: 64).

    Example:
        >>> generator = Generator(in_channels=1, out_channels=1)
        >>> noisy_mag = torch.randn(1, 1, 256, 256)
        >>> denoised_mag = generator(noisy_mag)
        >>> print(denoised_mag.shape)  # (1, 1, 256, 256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        bf = base_features

        # Encoder (downsampling path)
        # Each layer halves spatial dimensions
        # For 256x256 input: 256->128->64->32->16->8->4->2->1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, bf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )  # No BatchNorm on first layer
        self.enc2 = ConvBlock(bf, bf * 2)       # 64 -> 128
        self.enc3 = ConvBlock(bf * 2, bf * 4)   # 128 -> 256
        self.enc4 = ConvBlock(bf * 4, bf * 8)   # 256 -> 512
        self.enc5 = ConvBlock(bf * 8, bf * 8)   # 512 -> 512
        self.enc6 = ConvBlock(bf * 8, bf * 8)   # 512 -> 512
        self.enc7 = ConvBlock(bf * 8, bf * 8)   # 512 -> 512

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(bf * 8, bf * 8, 4, 2, 1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder (upsampling path) with skip connections
        # Input channels = decoder output + encoder skip
        self.dec1 = ConvTransposeBlock(bf * 8, bf * 8, use_dropout=True)
        self.dec2 = ConvTransposeBlock(bf * 8 * 2, bf * 8, use_dropout=True)
        self.dec3 = ConvTransposeBlock(bf * 8 * 2, bf * 8, use_dropout=True)
        self.dec4 = ConvTransposeBlock(bf * 8 * 2, bf * 8)
        self.dec5 = ConvTransposeBlock(bf * 8 * 2, bf * 4)
        self.dec6 = ConvTransposeBlock(bf * 4 * 2, bf * 2)
        self.dec7 = ConvTransposeBlock(bf * 2 * 2, bf)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights with normal distribution."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).
               Typically noisy magnetogram (256x256).

        Returns:
            Output tensor of shape (batch, out_channels, H, W).
            De-noised magnetogram.
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder with skip connections
        d1 = self.dec1(b)
        d1 = torch.cat([d1, e7], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e6], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e5], dim=1)

        d4 = self.dec4(d3)
        d4 = torch.cat([d4, e4], dim=1)

        d5 = self.dec5(d4)
        d5 = torch.cat([d5, e3], dim=1)

        d6 = self.dec6(d5)
        d6 = torch.cat([d6, e2], dim=1)

        d7 = self.dec7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        return self.final(d7)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.

    Classifies 70x70 overlapping patches as real or fake.
    Takes concatenated input (noisy + de-noised/target magnetogram).

    Args:
        in_channels: Number of input channels (noisy + output, default: 2).
        base_features: Base number of features (default: 64).

    Example:
        >>> discriminator = Discriminator(in_channels=2)
        >>> noisy = torch.randn(1, 1, 256, 256)
        >>> denoised = torch.randn(1, 1, 256, 256)
        >>> x = torch.cat([noisy, denoised], dim=1)
        >>> prob = discriminator(x)
        >>> print(prob.shape)  # (1, 1, 30, 30) for 256x256 input
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_features: int = 64,
    ) -> None:
        super().__init__()

        bf = base_features

        self.model = nn.Sequential(
            # Layer 1: No BatchNorm
            nn.Conv2d(in_channels, bf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(bf, bf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(bf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(bf * 2, bf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(bf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: stride=1
            nn.Conv2d(bf * 4, bf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(bf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output layer: stride=1
            nn.Conv2d(bf * 8, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights with normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Concatenated input tensor (noisy + output) of shape
               (batch, in_channels, H, W).

        Returns:
            Patch probabilities of shape (batch, 1, H', W').
        """
        return self.model(x)


class Pix2Pix(nn.Module):
    """
    Complete Pix2Pix model for magnetogram denoising.

    Combines Generator and Discriminator for conditional GAN training.
    Translates noisy magnetograms to de-noised equivalents.

    Args:
        in_channels: Number of input channels for generator.
        out_channels: Number of output channels for generator.
        lambda_l1: Weight for L1 loss (default: 100).

    Example:
        >>> model = Pix2Pix(in_channels=1, out_channels=1)
        >>> noisy = torch.randn(1, 1, 256, 256)
        >>> target = torch.randn(1, 1, 256, 256)  # stacked magnetogram
        >>> denoised, loss_g, loss_d = model(noisy, target)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        lambda_l1: float = 100.0,
    ) -> None:
        super().__init__()

        self.generator = Generator(in_channels, out_channels)
        self.discriminator = Discriminator(in_channels + out_channels)
        self.lambda_l1 = lambda_l1

        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()

    def forward(
        self,
        noisy_mag: torch.Tensor,
        target_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            noisy_mag: Input noisy magnetogram.
            target_mag: Target (stacked) magnetogram. If None, only generates.

        Returns:
            Tuple of (denoised_mag, generator_loss, discriminator_loss).
            Losses are None if target_mag is None (inference mode).
        """
        # Generate de-noised magnetogram
        denoised = self.generator(noisy_mag)

        if target_mag is None:
            return denoised, None, None

        # Discriminator forward
        real_pair = torch.cat([noisy_mag, target_mag], dim=1)
        fake_pair = torch.cat([noisy_mag, denoised.detach()], dim=1)

        pred_real = self.discriminator(real_pair)
        pred_fake = self.discriminator(fake_pair)

        # Discriminator loss
        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)

        loss_d_real = self.criterion_gan(pred_real, real_label)
        loss_d_fake = self.criterion_gan(pred_fake, fake_label)
        loss_d = (loss_d_real + loss_d_fake) * 0.5

        # Generator loss
        fake_pair_for_g = torch.cat([noisy_mag, denoised], dim=1)
        pred_fake_for_g = self.discriminator(fake_pair_for_g)

        loss_g_gan = self.criterion_gan(pred_fake_for_g, real_label)
        loss_g_l1 = self.criterion_l1(denoised, target_mag)
        loss_g = loss_g_gan + self.lambda_l1 * loss_g_l1

        return denoised, loss_g, loss_d


def get_generator(
    in_channels: int = 1,
    out_channels: int = 1,
) -> Generator:
    """
    Factory function to create Generator.

    Args:
        in_channels: Number of input channels (noisy magnetogram).
        out_channels: Number of output channels (de-noised magnetogram).

    Returns:
        Generator model.
    """
    return Generator(in_channels, out_channels)


def get_discriminator(
    in_channels: int = 2,
) -> Discriminator:
    """
    Factory function to create Discriminator.

    Args:
        in_channels: Number of input channels (noisy + output).

    Returns:
        Discriminator model.
    """
    return Discriminator(in_channels)


if __name__ == "__main__":
    # Test Generator
    print("Testing Generator...")
    generator = Generator(in_channels=1, out_channels=1)

    # Test with 256x256 input (standard size for this model)
    x = torch.randn(2, 1, 256, 256)
    y = generator(x)
    print(f"  Input shape: {x.shape} -> Output shape: {y.shape}")

    num_params_g = sum(p.numel() for p in generator.parameters())
    print(f"  Generator parameters: {num_params_g:,}")

    # Test Discriminator
    print("\nTesting Discriminator...")
    discriminator = Discriminator(in_channels=2)

    # Concatenate noisy and de-noised
    noisy = torch.randn(2, 1, 256, 256)
    denoised = torch.randn(2, 1, 256, 256)
    pair = torch.cat([noisy, denoised], dim=1)
    prob = discriminator(pair)
    print(f"  Input shape: {pair.shape} -> Output shape: {prob.shape}")

    num_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"  Discriminator parameters: {num_params_d:,}")

    # Test Pix2Pix
    print("\nTesting Pix2Pix...")
    model = Pix2Pix(in_channels=1, out_channels=1)

    noisy = torch.randn(2, 1, 256, 256)
    target = torch.randn(2, 1, 256, 256)
    denoised, loss_g, loss_d = model(noisy, target)
    print(f"  De-noised shape: {denoised.shape}")
    print(f"  Generator loss: {loss_g.item():.4f}")
    print(f"  Discriminator loss: {loss_d.item():.4f}")

    # Inference mode
    print("\nTesting inference mode...")
    denoised_only, _, _ = model(noisy, None)
    print(f"  De-noised output shape: {denoised_only.shape}")

    print("\nAll tests passed!")
