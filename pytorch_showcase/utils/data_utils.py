"""
Data loading and preprocessing utilities.

This module provides convenient functions for loading and preprocessing
datasets commonly used in deep learning projects.
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from pathlib import Path
import os


class TextDataset(Dataset):
    """Simple text dataset for demonstration purposes."""
    
    def __init__(self, texts: List[str], labels: List[int], 
                 vocab: Dict[str, int], max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
        
        # Add special tokens
        self.pad_token = vocab.get('<PAD>', 0)
        self.unk_token = vocab.get('<UNK>', 1)
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and encode
        tokens = text.lower().split()
        token_ids = [self.vocab.get(token, self.unk_token) for token in tokens]
        
        # Truncate or pad
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.pad_token] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def get_cifar10_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    val_split: float = 0.1,
    augment_train: bool = True,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get CIFAR-10 data loaders with train/validation/test splits.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        data_dir: Directory to store/load data
        val_split: Fraction of training data to use for validation
        augment_train: Whether to apply data augmentation to training set
        normalize: Whether to normalize images
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # CIFAR-10 statistics
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)
    
    # Training transforms
    train_transforms = []
    if augment_train:
        train_transforms.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    
    train_transforms.append(transforms.ToTensor())
    
    if normalize:
        train_transforms.append(transforms.Normalize(cifar10_mean, cifar10_std))
    
    train_transform = transforms.Compose(train_transforms)
    
    # Test transforms (no augmentation)
    test_transforms = [transforms.ToTensor()]
    if normalize:
        test_transforms.append(transforms.Normalize(cifar10_mean, cifar10_std))
    
    test_transform = transforms.Compose(test_transforms)
    
    # Download and load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform)
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform)
    
    # Split training data into train/validation
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))
        
        # Create validation dataset with test transforms (no augmentation)
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=test_transform)
        
        # Apply subset indices to validation dataset
        val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)
        
        train_dataset = train_subset
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def get_text_loaders(
    texts: List[str],
    labels: List[int],
    vocab: Optional[Dict[str, int]] = None,
    batch_size: int = 32,
    max_length: int = 128,
    val_split: float = 0.2,
    test_split: float = 0.1,
    min_freq: int = 2
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Create text data loaders from raw text data.
    
    Args:
        texts: List of text strings
        labels: List of corresponding labels
        vocab: Pre-built vocabulary (if None, will be created)
        batch_size: Batch size for data loaders
        max_length: Maximum sequence length
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
        min_freq: Minimum frequency for vocabulary inclusion
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, vocab)
    """
    
    # Build vocabulary if not provided
    if vocab is None:
        vocab = build_vocabulary(texts, min_freq=min_freq)
    
    # Create dataset
    dataset = TextDataset(texts, labels, vocab, max_length)
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, vocab


def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """
    Build vocabulary from text data.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for word inclusion
    
    Returns:
        Dictionary mapping words to indices
    """
    
    # Count word frequencies
    word_freq = {}
    for text in texts:
        words = text.lower().split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Build vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    
    # Add words that meet minimum frequency
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def create_sample_text_data(num_samples: int = 1000, num_classes: int = 3) -> Tuple[List[str], List[int]]:
    """
    Create sample text data for demonstration purposes.
    
    Args:
        num_samples: Number of text samples to generate
        num_classes: Number of classes
    
    Returns:
        Tuple of (texts, labels)
    """
    
    # Sample text templates for different classes
    templates = {
        0: [  # Positive sentiment
            "this movie is amazing and wonderful",
            "i love this film it's fantastic",
            "great acting and excellent story",
            "brilliant cinematography and direction",
            "outstanding performance by all actors"
        ],
        1: [  # Negative sentiment
            "this movie is terrible and boring",
            "i hate this film it's awful",
            "bad acting and poor story",
            "horrible cinematography and direction",
            "disappointing performance by all actors"
        ],
        2: [  # Neutral sentiment
            "this movie is okay and average",
            "the film is decent but not special",
            "acceptable acting and standard story",
            "normal cinematography and direction",
            "adequate performance by most actors"
        ]
    }
    
    texts = []
    labels = []
    
    np.random.seed(42)
    
    for _ in range(num_samples):
        label = np.random.randint(0, min(num_classes, len(templates)))
        template = np.random.choice(templates[label])
        
        # Add some variation
        words = template.split()
        if np.random.random() < 0.3:  # 30% chance to add variation
            # Randomly shuffle some words or add noise
            if len(words) > 3:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        text = ' '.join(words)
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def get_mnist_loaders(
    batch_size: int = 32,
    num_workers: int = 4,
    data_dir: str = "./data",
    val_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MNIST data loaders for GAN training.
    
    Args:
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
        data_dir: Directory to store/load data
        val_split: Fraction of training data for validation
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Transforms for MNIST
    transform = transforms.Compose([
        transforms.Resize(64),  # Resize to 64x64 for GAN
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])
    
    # Download and load datasets
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform)
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform)
    
    # Split training data
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42))
    else:
        val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True) if val_dataset else None
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize CIFAR-10 images for visualization."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    return tensor * std + mean


def get_class_names(dataset: str) -> List[str]:
    """Get class names for different datasets."""
    
    class_names = {
        'cifar10': [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ],
        'mnist': [str(i) for i in range(10)],
        'text_sentiment': ['negative', 'neutral', 'positive']
    }
    
    return class_names.get(dataset.lower(), [])


# Data augmentation utilities
class MixUp:
    """MixUp data augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply MixUp augmentation."""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix data augmentation."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix augmentation."""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        y_a, y_b = y, y[index]
        
        # Generate random bounding box
        W, H = x.size(2), x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return x, y_a, y_b, lam