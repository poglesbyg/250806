"""Models package containing various neural network architectures."""

from .cnn import CIFAR10CNN
from .transformer import TextClassifier
from .gan import SimpleGAN

__all__ = ["CIFAR10CNN", "TextClassifier", "SimpleGAN"]