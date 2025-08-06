"""
Visualization utilities for models, data, and training progress.

This module provides functions for visualizing:
- Model architectures and feature maps
- Data samples and augmentations
- Training progress and metrics
- Generated samples from GANs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path


# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelVisualizer:
    """
    Comprehensive model visualization utility.
    
    Provides methods for visualizing:
    - Model architectures
    - Feature maps and activations
    - Attention weights
    - Gradients and saliency maps
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}
        self.hooks = []
    
    def register_hooks(self, layer_names: List[str]) -> None:
        """Register forward and backward hooks for specified layers."""
        
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(grad):
                self.gradients[name] = grad.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if name in layer_names:
                # Forward hook
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)
                
                # Backward hook for gradients
                if hasattr(module, 'weight') and module.weight is not None:
                    handle = module.weight.register_hook(get_gradient(name))
                    self.hooks.append(handle)
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def visualize_feature_maps(self, input_tensor: torch.Tensor, 
                              layer_name: str,
                              num_features: int = 16,
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize feature maps from a specific layer.
        
        Args:
            input_tensor: Input to the model
            layer_name: Name of layer to visualize
            num_features: Number of feature maps to show
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        
        self.model.eval()
        
        # Register hook for the specified layer
        self.register_hooks([layer_name])
        
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))
        
        if layer_name not in self.activations:
            raise ValueError(f"Layer '{layer_name}' not found or no activations captured")
        
        feature_maps = self.activations[layer_name]
        
        # Take first sample from batch
        if feature_maps.dim() == 4:  # Conv layer: [B, C, H, W]
            feature_maps = feature_maps[0]  # [C, H, W]
            num_channels = min(num_features, feature_maps.size(0))
            
            # Create subplot grid
            cols = 4
            rows = (num_channels + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_channels):
                row, col = i // cols, i % cols
                
                feature_map = feature_maps[i].cpu().numpy()
                
                im = axes[row, col].imshow(feature_map, cmap='viridis')
                axes[row, col].set_title(f'Feature {i}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col])
            
            # Hide unused subplots
            for i in range(num_channels, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
        
        else:  # Fully connected layer
            feature_maps = feature_maps[0].cpu().numpy()  # [Features]
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.bar(range(len(feature_maps)), feature_maps)
            ax.set_title(f'Activations in {layer_name}')
            ax.set_xlabel('Neuron')
            ax.set_ylabel('Activation')
        
        plt.suptitle(f'Feature Maps from {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        self.remove_hooks()
        return fig
    
    def visualize_filters(self, layer_name: str,
                         num_filters: int = 16,
                         figsize: Tuple[int, int] = (12, 8),
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize convolutional filters from a specific layer.
        
        Args:
            layer_name: Name of convolutional layer
            num_filters: Number of filters to visualize
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        
        # Find the layer
        layer = None
        for name, module in self.model.named_modules():
            if name == layer_name and isinstance(module, (nn.Conv2d, nn.Conv1d)):
                layer = module
                break
        
        if layer is None:
            raise ValueError(f"Convolutional layer '{layer_name}' not found")
        
        weights = layer.weight.data.cpu()
        
        if weights.dim() == 4:  # Conv2d: [out_channels, in_channels, H, W]
            num_filters = min(num_filters, weights.size(0))
            
            # If multiple input channels, show only the first one or create RGB
            if weights.size(1) == 3:  # RGB input
                filters_to_show = weights[:num_filters]  # [num_filters, 3, H, W]
            else:
                filters_to_show = weights[:num_filters, 0:1]  # [num_filters, 1, H, W]
            
            # Normalize filters for visualization
            filters_to_show = (filters_to_show - filters_to_show.min()) / (filters_to_show.max() - filters_to_show.min())
            
            # Create grid
            cols = 4
            rows = (num_filters + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if rows == 1:
                axes = axes.reshape(1, -1)
            
            for i in range(num_filters):
                row, col = i // cols, i % cols
                
                filter_img = filters_to_show[i]
                
                if filter_img.size(0) == 3:  # RGB
                    filter_img = filter_img.permute(1, 2, 0)
                    axes[row, col].imshow(filter_img)
                else:  # Grayscale
                    axes[row, col].imshow(filter_img.squeeze(), cmap='gray')
                
                axes[row, col].set_title(f'Filter {i}')
                axes[row, col].axis('off')
            
            # Hide unused subplots
            for i in range(num_filters, rows * cols):
                row, col = i // cols, i % cols
                axes[row, col].axis('off')
        
        plt.suptitle(f'Filters from {layer_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_weights(self, input_tensor: torch.Tensor,
                                   attention_weights: torch.Tensor,
                                   input_tokens: Optional[List[str]] = None,
                                   figsize: Tuple[int, int] = (10, 8),
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize attention weights from transformer models.
        
        Args:
            input_tensor: Input tensor
            attention_weights: Attention weights [batch, heads, seq_len, seq_len]
            input_tokens: List of input tokens for labeling
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        
        # Take first sample and average over heads
        if attention_weights.dim() == 4:
            attn = attention_weights[0].mean(0).cpu().numpy()  # [seq_len, seq_len]
        else:
            attn = attention_weights.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(attn, annot=False, cmap='Blues', ax=ax)
        
        ax.set_title('Attention Weights')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        
        # Add token labels if provided
        if input_tokens:
            ax.set_xticklabels(input_tokens, rotation=45, ha='right')
            ax.set_yticklabels(input_tokens, rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_grad_cam(self, input_tensor: torch.Tensor, 
                        target_class: int,
                        target_layer: str) -> torch.Tensor:
        """
        Compute Grad-CAM for a specific class and layer.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            target_layer: Name of target layer
        
        Returns:
            Grad-CAM heatmap
        """
        
        self.model.eval()
        self.register_hooks([target_layer])
        
        # Forward pass
        input_tensor.requires_grad_()
        logits = self.model(input_tensor.to(self.device))
        
        # Backward pass
        self.model.zero_grad()
        class_score = logits[0, target_class]
        class_score.backward()
        
        # Get activations and gradients
        activations = self.activations[target_layer]
        gradients = self.gradients.get(target_layer)
        
        if gradients is None:
            # If no gradients captured, compute them manually
            gradients = torch.autograd.grad(class_score, activations, retain_graph=True)[0]
        
        # Compute weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        
        # Compute Grad-CAM
        grad_cam = (weights * activations).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        
        # Normalize
        grad_cam = grad_cam.squeeze()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        
        self.remove_hooks()
        return grad_cam.detach().cpu()
    
    def visualize_grad_cam(self, input_tensor: torch.Tensor,
                          target_class: int,
                          target_layer: str,
                          original_image: Optional[torch.Tensor] = None,
                          figsize: Tuple[int, int] = (12, 4),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize Grad-CAM heatmap overlaid on original image.
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class index
            target_layer: Name of target layer
            original_image: Original image (before preprocessing)
            figsize: Figure size
            save_path: Path to save the plot
        
        Returns:
            Matplotlib figure
        """
        
        grad_cam = self.compute_grad_cam(input_tensor, target_class, target_layer)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if original_image is not None:
            img = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
        else:
            img = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            img = (img - img.min()) / (img.max() - img.min())
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grad-CAM heatmap
        axes[1].imshow(grad_cam, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img)
        axes[2].imshow(grad_cam, cmap='jet', alpha=0.4)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'Grad-CAM for Class {target_class}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def visualize_data_samples(dataloader: torch.utils.data.DataLoader,
                          class_names: Optional[List[str]] = None,
                          num_samples: int = 16,
                          figsize: Tuple[int, int] = (12, 8),
                          denormalize_fn: Optional[callable] = None,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize sample images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader
        class_names: List of class names
        num_samples: Number of samples to show
        figsize: Figure size
        denormalize_fn: Function to denormalize images
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    # Get a batch of data
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    # Select subset
    num_samples = min(num_samples, len(images))
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # Denormalize if function provided
    if denormalize_fn:
        images = denormalize_fn(images)
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row, col = i // cols, i % cols
        
        img = images[i]
        
        # Handle different image formats
        if img.dim() == 3:
            if img.size(0) == 3:  # RGB
                img = img.permute(1, 2, 0)
                axes[row, col].imshow(img.clamp(0, 1))
            else:  # Grayscale
                axes[row, col].imshow(img.squeeze(), cmap='gray')
        else:  # Already in HWC format
            axes[row, col].imshow(img, cmap='gray' if img.shape[-1] == 1 else None)
        
        # Add title with class name
        label = labels[i].item()
        title = class_names[label] if class_names else f'Class {label}'
        axes[row, col].set_title(title)
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Data Samples', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_generated_samples(samples: torch.Tensor,
                               num_samples: int = 16,
                               figsize: Tuple[int, int] = (12, 8),
                               title: str = "Generated Samples",
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize generated samples from GANs.
    
    Args:
        samples: Generated image samples
        num_samples: Number of samples to show
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    num_samples = min(num_samples, len(samples))
    samples = samples[:num_samples]
    
    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    
    # Create grid
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        row, col = i // cols, i % cols
        
        img = samples[i]
        
        if img.size(0) == 3:  # RGB
            img = img.permute(1, 2, 0)
            axes[row, col].imshow(img)
        else:  # Grayscale
            axes[row, col].imshow(img.squeeze(), cmap='gray')
        
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_model_summary(model: nn.Module, input_size: Tuple[int, ...],
                        device: torch.device) -> str:
    """
    Create a detailed summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (without batch dimension)
        device: Device to run the model on
    
    Returns:
        String containing model summary
    """
    
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            
            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = {}
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = -1  # batch size
            
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = -1
            
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params
        
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))
    
    # Create summary dict
    summary = {}
    hooks = []
    
    # Register hooks
    model.apply(register_hook)
    
    # Make a forward pass
    model.eval()
    
    # Check if this is a text model (has embedding layer) - use integer input
    embedding_layers = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if embedding_layers and len(input_size) == 1:
        # Text model - use random integers within vocabulary size
        vocab_size = embedding_layers[0].num_embeddings
        x = torch.randint(0, vocab_size, (1, *input_size)).to(device)
    else:
        # Vision model - use random floats
        x = torch.randn(1, *input_size).to(device)
    
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Create summary string
    summary_str = "Model Summary\n"
    summary_str += "=" * 80 + "\n"
    summary_str += f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15} {'Trainable':<10}\n"
    summary_str += "=" * 80 + "\n"
    
    total_params = 0
    trainable_params = 0
    
    for layer in summary:
        line_str = f"{layer:<25} "
        
        output_shape = summary[layer]["output_shape"]
        if isinstance(output_shape[0], list):
            shape_str = str(output_shape)
        else:
            shape_str = str(output_shape)
        line_str += f"{shape_str:<20} "
        
        num_params = summary[layer]["nb_params"]
        line_str += f"{num_params:,}".ljust(15) + " "
        
        trainable = summary[layer].get("trainable", "N/A")
        line_str += f"{str(trainable):<10}\n"
        
        summary_str += line_str
        
        total_params += num_params
        if trainable:
            trainable_params += num_params
    
    summary_str += "=" * 80 + "\n"
    summary_str += f"Total params: {total_params:,}\n"
    summary_str += f"Trainable params: {trainable_params:,}\n"
    summary_str += f"Non-trainable params: {total_params - trainable_params:,}\n"
    summary_str += "=" * 80 + "\n"
    
    return summary_str


def plot_model_architecture(model: nn.Module, save_path: Optional[str] = None) -> str:
    """
    Create a text-based visualization of model architecture.
    
    Args:
        model: PyTorch model
        save_path: Path to save the architecture diagram
    
    Returns:
        String representation of the architecture
    """
    
    arch_str = "Model Architecture\n"
    arch_str += "=" * 50 + "\n"
    
    def add_layer_info(module, prefix=""):
        nonlocal arch_str
        
        for name, child in module.named_children():
            layer_name = f"{prefix}{name}"
            layer_type = child.__class__.__name__
            
            # Get layer parameters
            params = []
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                params.append(f"in={child.in_features}, out={child.out_features}")
            if hasattr(child, 'in_channels') and hasattr(child, 'out_channels'):
                params.append(f"in={child.in_channels}, out={child.out_channels}")
            if hasattr(child, 'kernel_size'):
                params.append(f"kernel={child.kernel_size}")
            if hasattr(child, 'stride'):
                params.append(f"stride={child.stride}")
            if hasattr(child, 'padding'):
                params.append(f"padding={child.padding}")
            
            param_str = f"({', '.join(params)})" if params else ""
            
            arch_str += f"{prefix}├── {name}: {layer_type}{param_str}\n"
            
            # Recursively add children
            if list(child.children()):
                add_layer_info(child, prefix + "│   ")
    
    add_layer_info(model)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(arch_str)
    
    return arch_str