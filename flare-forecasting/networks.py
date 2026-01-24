"""
CNN Models for Solar Flare Forecasting.

This module implements the CNN models from:
Park et al. (2018), "Application of the Deep Convolutional Neural Network
to the Forecast of Solar Flare Occurrence Using Full-disk Solar Magnetograms",
ApJ, 869, 91.
https://doi.org/10.3847/1538-4357/aaed40

Three models are provided:
- Model 1: AlexNet-style
- Model 2: GoogLeNet-style (simplified)
- Model 3: Proposed model (GoogLeNet + DenseNet combination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ConvBNReLU(nn.Module):
    """
    Convolution + Batch Normalization + ReLU block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding. If None, computed automatically.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 3,
        stride: int = 1,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            else:
                padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    """
    Inception-style convolution block with multiple filter sizes.

    This block applies 1x1, 1x3, 3x1, and 3x3 convolutions in parallel,
    then concatenates the results.

    Args:
        in_channels: Number of input channels.
        growth_rate: Number of output channels for each branch.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
    ) -> None:
        super().__init__()

        r = growth_rate

        # First stage: parallel convolutions
        self.conv1x1 = ConvBNReLU(in_channels, r, kernel_size=1)
        self.conv1x3 = ConvBNReLU(in_channels, r, kernel_size=(1, 3))
        self.conv3x1 = ConvBNReLU(in_channels, r, kernel_size=(3, 1))
        self.conv3x3 = ConvBNReLU(in_channels, r, kernel_size=3)

        # Second stage: after concatenation (4*r channels)
        concat_channels = 4 * r
        self.conv1x1_2 = ConvBNReLU(concat_channels, concat_channels, kernel_size=1)
        self.conv3x3_2 = ConvBNReLU(concat_channels, r, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First stage
        out1x1 = self.conv1x1(x)
        out1x3 = self.conv1x3(x)
        out3x1 = self.conv3x1(x)
        out3x3 = self.conv3x3(x)

        # Concatenate first stage outputs
        concat1 = torch.cat([out1x1, out1x3, out3x1, out3x3], dim=1)

        # Second stage
        out1x1_2 = self.conv1x1_2(concat1)
        out3x3_2 = self.conv3x3_2(concat1)

        # Final concatenation
        return torch.cat([out1x1_2, out3x3_2], dim=1)


class DenseBlock(nn.Module):
    """
    Dense block with inception-style convolutions.

    Each convolution block is connected to all subsequent blocks
    (DenseNet-style dense connectivity).

    Args:
        in_channels: Number of input channels.
        growth_rate: Growth rate (feature increase per block).
        num_blocks: Number of convolution blocks in this dense block.
    """

    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList()

        # Each block outputs (4*r + r) = 5*r new channels
        # But based on the paper figure, output is (2k + r) where k grows
        # Simplified: each block adds growth_rate channels
        current_channels = in_channels

        for i in range(num_blocks):
            # Simplified inception block
            block = nn.Sequential(
                ConvBNReLU(current_channels, growth_rate, kernel_size=1),
                ConvBNReLU(growth_rate, growth_rate, kernel_size=3),
            )
            self.blocks.append(block)
            current_channels += growth_rate

        self.out_channels = current_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = [x]

        for block in self.blocks:
            # Concatenate all previous features (dense connectivity)
            concat_features = torch.cat(features, dim=1)
            new_features = block(concat_features)
            features.append(new_features)

        return torch.cat(features, dim=1)


class TransitionLayer(nn.Module):
    """
    Transition layer between dense blocks.

    Reduces spatial dimensions and optionally reduces channels.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class FlarePredictor(nn.Module):
    """
    Proposed CNN model for solar flare prediction.

    Combines GoogLeNet's inception-style convolutions with DenseNet's
    dense connectivity for effective feature extraction from full-disk
    magnetograms.

    Args:
        in_channels: Number of input channels (default: 1 for magnetogram).
        num_classes: Number of output classes (default: 2 for binary).
        growth_rate: Growth rate for dense blocks (default: 16).
        num_modules: Number of dense modules (default: 4).
        blocks_per_module: Number of blocks per module (default: 6).
        init_features: Initial number of features (default: 16).

    Example:
        >>> model = FlarePredictor()
        >>> x = torch.randn(1, 1, 1024, 1024)
        >>> output = model(x)
        >>> print(output.shape)  # (1, 2)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        growth_rate: int = 16,
        num_modules: int = 4,
        blocks_per_module: int = 6,
        init_features: int = 16,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Build dense modules with transition layers
        self.modules_list = nn.ModuleList()
        self.transitions = nn.ModuleList()

        current_channels = init_features

        for i in range(num_modules):
            # Dense block
            dense_block = DenseBlock(
                in_channels=current_channels,
                growth_rate=growth_rate,
                num_blocks=blocks_per_module,
            )
            self.modules_list.append(dense_block)
            current_channels = dense_block.out_channels

            # Transition layer (except for the last module)
            if i < num_modules - 1:
                out_channels = current_channels // 2
                transition = TransitionLayer(current_channels, out_channels)
                self.transitions.append(transition)
                current_channels = out_channels

        # Final batch norm
        self.final_bn = nn.BatchNorm2d(current_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Output tensor of shape (batch, num_classes).
        """
        # Initial convolution
        x = self.init_conv(x)

        # Dense modules with transitions
        for i, dense_block in enumerate(self.modules_list):
            x = dense_block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # Final normalization and classification
        x = self.final_bn(x)
        x = F.relu(x)
        x = self.classifier(x)

        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict flare occurrence.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Predicted class (0: No-flare, 1: Flare).
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict flare probability.

        Args:
            x: Input tensor of shape (batch, in_channels, H, W).

        Returns:
            Probabilities of shape (batch, num_classes).
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class AlexNetFlare(nn.Module):
    """
    AlexNet-style model for flare prediction (Model 1).

    Simplified AlexNet architecture adapted for single-channel
    magnetogram input.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception module for GoogLeNet-style model.

    Args:
        in_channels: Number of input channels.
        ch1x1: Number of 1x1 conv output channels.
        ch3x3_reduce: Number of 1x1 conv channels before 3x3.
        ch3x3: Number of 3x3 conv output channels.
        ch5x5_reduce: Number of 1x1 conv channels before 5x5.
        ch5x5: Number of 5x5 conv output channels.
        pool_proj: Number of 1x1 conv channels after pooling.
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()

        # 1x1 branch
        self.branch1 = ConvBNReLU(in_channels, ch1x1, kernel_size=1)

        # 1x1 -> 3x3 branch
        self.branch2 = nn.Sequential(
            ConvBNReLU(in_channels, ch3x3_reduce, kernel_size=1),
            ConvBNReLU(ch3x3_reduce, ch3x3, kernel_size=3),
        )

        # 1x1 -> 5x5 branch
        self.branch3 = nn.Sequential(
            ConvBNReLU(in_channels, ch5x5_reduce, kernel_size=1),
            ConvBNReLU(ch5x5_reduce, ch5x5, kernel_size=5),
        )

        # pool -> 1x1 branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBNReLU(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)


class GoogLeNetFlare(nn.Module):
    """
    GoogLeNet-style model for flare prediction (Model 2).

    Simplified GoogLeNet with inception modules.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=1),
            ConvBNReLU(64, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Inception modules
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_model(
    model_type: str = "proposed",
    in_channels: int = 1,
    num_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create flare prediction model.

    Args:
        model_type: Type of model ("alexnet", "googlenet", "proposed").
        in_channels: Number of input channels.
        num_classes: Number of output classes.
        **kwargs: Additional arguments for the model.

    Returns:
        Instantiated model.
    """
    if model_type.lower() == "alexnet":
        return AlexNetFlare(in_channels, num_classes)
    elif model_type.lower() == "googlenet":
        return GoogLeNetFlare(in_channels, num_classes)
    elif model_type.lower() == "proposed":
        return FlarePredictor(in_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test proposed model
    print("Testing FlarePredictor (Proposed Model)...")
    model = FlarePredictor(
        in_channels=1,
        num_classes=2,
        growth_rate=16,
        num_modules=4,
        blocks_per_module=6,
    )

    # Test with smaller input for speed
    x = torch.randn(2, 1, 256, 256)
    output = model(x)
    print(f"  Input shape: {x.shape} -> Output shape: {output.shape}")

    probs = model.predict_proba(x)
    print(f"  Probabilities shape: {probs.shape}")

    preds = model.predict(x)
    print(f"  Predictions: {preds}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test AlexNet
    print("\nTesting AlexNetFlare...")
    alexnet = AlexNetFlare()
    output_alex = alexnet(x)
    print(f"  Input shape: {x.shape} -> Output shape: {output_alex.shape}")

    num_params_alex = sum(p.numel() for p in alexnet.parameters())
    print(f"  Total parameters: {num_params_alex:,}")

    # Test GoogLeNet
    print("\nTesting GoogLeNetFlare...")
    googlenet = GoogLeNetFlare()
    output_google = googlenet(x)
    print(f"  Input shape: {x.shape} -> Output shape: {output_google.shape}")

    num_params_google = sum(p.numel() for p in googlenet.parameters())
    print(f"  Total parameters: {num_params_google:,}")

    print("\nAll tests passed!")
