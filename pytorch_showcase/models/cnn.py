"""
Convolutional Neural Network for CIFAR-10 image classification.

This module implements a modern CNN architecture with:
- Batch normalization
- Dropout for regularization
- Residual connections
- Adaptive average pooling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualBlock(nn.Module):
    """A residual block with batch normalization and skip connections."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = F.relu(out)
        return out


class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 classification with residual connections.
    
    Architecture:
    - Initial conv layer
    - 3 residual blocks with increasing channels
    - Global average pooling
    - Final classification layer
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using He initialization."""
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
        """Forward pass through the network."""
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Extract feature maps from a specific layer for visualization."""
        x = F.relu(self.bn1(self.conv1(x)))
        
        if layer_name == 'layer1':
            return self.layer1(x)
        
        x = self.layer1(x)
        if layer_name == 'layer2':
            return self.layer2(x)
        
        x = self.layer2(x)
        if layer_name == 'layer3':
            return self.layer3(x)
        
        raise ValueError(f"Unknown layer name: {layer_name}")


def create_cifar10_cnn(pretrained: bool = False, **kwargs) -> CIFAR10CNN:
    """Create a CIFAR-10 CNN model."""
    model = CIFAR10CNN(**kwargs)
    
    if pretrained:
        # In a real scenario, you would load pretrained weights here
        print("Note: Pretrained weights not available for this custom model")
    
    return model