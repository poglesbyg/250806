"""
Generative Adversarial Network (GAN) for image generation.

This module implements a DCGAN (Deep Convolutional GAN) with:
- Generator network with transposed convolutions
- Discriminator network with convolutional layers
- Batch normalization and LeakyReLU activations
- Spectral normalization for training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


def weights_init(m: nn.Module) -> None:
    """Initialize network weights according to DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """
    Generator network that creates images from random noise.
    
    Architecture:
    - Input: Random noise vector (latent_dim)
    - Series of transposed convolutions with batch norm and ReLU
    - Output: Generated image (3 x 64 x 64)
    """
    
    def __init__(self, latent_dim: int = 100, num_channels: int = 3, 
                 feature_map_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Initial projection and reshape
        self.initial = nn.Sequential(
            nn.Linear(latent_dim, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Transposed convolution layers
        self.conv_layers = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_map_size, num_channels, 
                              kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        self.feature_map_size = feature_map_size
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise vectors."""
        batch_size = z.size(0)
        
        # Project and reshape
        x = self.initial(z)
        x = x.view(batch_size, self.feature_map_size * 8, 4, 4)
        
        # Generate image
        x = self.conv_layers(x)
        
        return x
    
    def generate_samples(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate random samples."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            samples = self(z)
        return samples


class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or fake.
    
    Architecture:
    - Input: Image (3 x 64 x 64)
    - Series of convolutions with batch norm and LeakyReLU
    - Output: Single value (real/fake probability)
    """
    
    def __init__(self, num_channels: int = 3, feature_map_size: int = 64,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        def conv_block(in_channels: int, out_channels: int, kernel_size: int = 4,
                      stride: int = 2, padding: int = 1, normalize: bool = True) -> list:
            """Create a convolutional block."""
            layers = []
            
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            return layers
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 64x64 -> 32x32
            *conv_block(num_channels, feature_map_size, normalize=False),
            
            # 32x32 -> 16x16
            *conv_block(feature_map_size, feature_map_size * 2),
            
            # 16x16 -> 8x8
            *conv_block(feature_map_size * 2, feature_map_size * 4),
            
            # 8x8 -> 4x4
            *conv_block(feature_map_size * 4, feature_map_size * 8),
            
            # 4x4 -> 1x1
            nn.Conv2d(feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake."""
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)


class SimpleGAN(nn.Module):
    """
    Complete GAN model combining Generator and Discriminator.
    
    This class provides a convenient interface for training and generation.
    """
    
    def __init__(self, latent_dim: int = 100, num_channels: int = 3,
                 feature_map_size: int = 64, use_spectral_norm: bool = True):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, num_channels, feature_map_size)
        self.discriminator = Discriminator(num_channels, feature_map_size, use_spectral_norm)
        
        # Loss function
        self.criterion = nn.BCELoss()
    
    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate samples using the generator."""
        return self.generator.generate_samples(num_samples, device)
    
    def discriminate(self, images: torch.Tensor) -> torch.Tensor:
        """Classify images using the discriminator."""
        return self.discriminator(images)
    
    def generator_loss(self, fake_images: torch.Tensor) -> torch.Tensor:
        """Calculate generator loss (wants discriminator to classify fake as real)."""
        fake_labels = torch.ones(fake_images.size(0), 1, device=fake_images.device)
        fake_output = self.discriminator(fake_images)
        return self.criterion(fake_output, fake_labels)
    
    def discriminator_loss(self, real_images: torch.Tensor, 
                          fake_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate discriminator loss (wants to classify real as real, fake as fake)."""
        batch_size = real_images.size(0)
        device = real_images.device
        
        # Real images
        real_labels = torch.ones(batch_size, 1, device=device)
        real_output = self.discriminator(real_images)
        real_loss = self.criterion(real_output, real_labels)
        
        # Fake images
        fake_labels = torch.zeros(batch_size, 1, device=device)
        fake_output = self.discriminator(fake_images.detach())
        fake_loss = self.criterion(fake_output, fake_labels)
        
        total_loss = real_loss + fake_loss
        return total_loss, real_loss, fake_loss
    
    def get_generator_parameters(self):
        """Get generator parameters for optimizer."""
        return self.generator.parameters()
    
    def get_discriminator_parameters(self):
        """Get discriminator parameters for optimizer."""
        return self.discriminator.parameters()


class ConditionalGAN(nn.Module):
    """
    Conditional GAN that can generate images conditioned on class labels.
    
    This extends the basic GAN to support conditional generation.
    """
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10,
                 num_channels: int = 3, feature_map_size: int = 64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Modified generator that takes concatenated input
        self.generator = Generator(latent_dim * 2, num_channels, feature_map_size)
        
        # Modified discriminator with additional input for labels
        self.discriminator = self._create_conditional_discriminator(
            num_channels, num_classes, feature_map_size)
        
        self.criterion = nn.BCELoss()
    
    def _create_conditional_discriminator(self, num_channels: int, 
                                        num_classes: int, feature_map_size: int) -> nn.Module:
        """Create a discriminator that takes both images and labels."""
        
        class ConditionalDiscriminator(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Label embedding
                self.label_embedding = nn.Embedding(num_classes, 64 * 64)
                
                # Convolutional layers
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(num_channels + 1, feature_map_size, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature_map_size * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature_map_size * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(feature_map_size * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid()
                )
                
                self.apply(weights_init)
            
            def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                # Embed labels and reshape to image format
                label_embedding = self.label_embedding(labels)
                label_embedding = label_embedding.view(-1, 1, 64, 64)
                
                # Concatenate images and label embeddings
                x = torch.cat([images, label_embedding], dim=1)
                
                # Pass through convolutional layers
                output = self.conv_layers(x)
                return output.view(output.size(0), -1)
        
        return ConditionalDiscriminator()
    
    def generate(self, num_samples: int, labels: torch.Tensor, 
                device: torch.device) -> torch.Tensor:
        """Generate samples conditioned on labels."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        label_embeddings = self.label_embedding(labels)
        
        # Concatenate noise and label embeddings
        conditional_input = torch.cat([z, label_embeddings], dim=1)
        
        with torch.no_grad():
            samples = self.generator(conditional_input)
        
        return samples


def create_gan(gan_type: str = "simple", **kwargs) -> nn.Module:
    """Factory function to create different types of GANs."""
    
    if gan_type == "simple":
        return SimpleGAN(**kwargs)
    elif gan_type == "conditional":
        return ConditionalGAN(**kwargs)
    else:
        raise ValueError(f"Unknown GAN type: {gan_type}")


# Utility functions for GAN training
def gradient_penalty(discriminator: nn.Module, real_samples: torch.Tensor,
                    fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP."""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty