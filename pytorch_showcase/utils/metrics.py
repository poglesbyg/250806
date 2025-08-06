"""
Metrics and evaluation utilities for different types of models.

This module provides functions for computing various metrics and
creating visualizations for model evaluation.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def compute_classification_metrics(predictions: torch.Tensor, 
                                 targets: torch.Tensor,
                                 num_classes: Optional[int] = None) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: True labels
        num_classes: Number of classes (inferred if not provided)
    
    Returns:
        Dictionary containing various metrics
    """
    
    # Convert to numpy
    if torch.is_tensor(predictions):
        if predictions.dim() > 1:
            pred_classes = predictions.argmax(dim=1).cpu().numpy()
            pred_probs = F.softmax(predictions, dim=1).cpu().numpy()
        else:
            pred_classes = predictions.cpu().numpy()
            pred_probs = None
    else:
        pred_classes = predictions
        pred_probs = None
    
    if torch.is_tensor(targets):
        true_labels = targets.cpu().numpy()
    else:
        true_labels = targets
    
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(true_labels, pred_classes)
    
    # Precision, recall, F1-score
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_classes, average='weighted', zero_division=0)
    
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        true_labels, pred_classes, average=None, zero_division=0)
    
    if num_classes is None:
        num_classes = len(np.unique(true_labels))
    
    for i in range(min(num_classes, len(precision_per_class))):
        metrics[f'precision_class_{i}'] = precision_per_class[i]
        metrics[f'recall_class_{i}'] = recall_per_class[i]
        metrics[f'f1_class_{i}'] = f1_per_class[i]
    
    # AUC-ROC for multi-class (if probabilities available)
    if pred_probs is not None and pred_probs.shape[1] > 2:
        try:
            # One-vs-rest AUC
            auc_scores = []
            for i in range(pred_probs.shape[1]):
                binary_labels = (true_labels == i).astype(int)
                if len(np.unique(binary_labels)) > 1:  # Avoid single-class issues
                    auc = roc_auc_score(binary_labels, pred_probs[:, i])
                    auc_scores.append(auc)
            
            if auc_scores:
                metrics['auc_macro'] = np.mean(auc_scores)
        except ValueError as e:
            warnings.warn(f"Could not compute AUC: {e}")
    
    return metrics


def compute_regression_metrics(predictions: torch.Tensor, 
                             targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        predictions: Model predictions
        targets: True values
    
    Returns:
        Dictionary containing regression metrics
    """
    
    # Convert to numpy
    if torch.is_tensor(predictions):
        pred_vals = predictions.cpu().numpy().flatten()
    else:
        pred_vals = predictions.flatten()
    
    if torch.is_tensor(targets):
        true_vals = targets.cpu().numpy().flatten()
    else:
        true_vals = targets.flatten()
    
    metrics = {}
    
    # Mean Squared Error
    mse = np.mean((pred_vals - true_vals) ** 2)
    metrics['mse'] = mse
    metrics['rmse'] = np.sqrt(mse)
    
    # Mean Absolute Error
    metrics['mae'] = np.mean(np.abs(pred_vals - true_vals))
    
    # R² Score
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    metrics['r2_score'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Mean Absolute Percentage Error
    non_zero_mask = true_vals != 0
    if np.any(non_zero_mask):
        mape = np.mean(np.abs((true_vals[non_zero_mask] - pred_vals[non_zero_mask]) / true_vals[non_zero_mask])) * 100
        metrics['mape'] = mape
    
    return metrics


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                   task_type: str = "classification", **kwargs) -> Dict[str, float]:
    """
    Compute metrics based on task type.
    
    Args:
        predictions: Model predictions
        targets: True labels/values
        task_type: Either "classification" or "regression"
        **kwargs: Additional arguments for specific metric functions
    
    Returns:
        Dictionary containing computed metrics
    """
    
    if task_type.lower() == "classification":
        return compute_classification_metrics(predictions, targets, **kwargs)
    elif task_type.lower() == "regression":
        return compute_regression_metrics(predictions, targets, **kwargs)
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         normalize: bool = False, 
                         title: str = "Confusion Matrix",
                         figsize: Tuple[int, int] = (8, 6),
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize the matrix
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(y_true: np.ndarray, y_scores: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   title: str = "ROC Curves",
                   figsize: Tuple[int, int] = (10, 8),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot ROC curves for multi-class classification.
    
    Args:
        y_true: True labels (one-hot encoded or integers)
        y_scores: Prediction scores/probabilities
        class_names: Names of classes
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle different input formats
    if y_scores.ndim == 1:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    else:
        # Multi-class classification
        n_classes = y_scores.shape[1]
        
        # Convert to one-hot if needed
        if y_true.ndim == 1:
            y_true_onehot = np.zeros((len(y_true), n_classes))
            y_true_onehot[np.arange(len(y_true)), y_true] = 1
            y_true = y_true_onehot
        
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            auc = roc_auc_score(y_true[:, i], y_scores[:, i])
            
            class_name = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        train_metrics: Optional[List[Dict[str, float]]] = None,
                        val_metrics: Optional[List[Dict[str, float]]] = None,
                        metric_name: str = "accuracy",
                        title: str = "Training Curves",
                        figsize: Tuple[int, int] = (12, 5),
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: Training losses over epochs
        val_losses: Validation losses over epochs
        train_metrics: Training metrics over epochs
        val_metrics: Validation metrics over epochs
        metric_name: Name of metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics if available
    if train_metrics and val_metrics and metric_name:
        train_metric_vals = [m.get(metric_name, 0) for m in train_metrics]
        val_metric_vals = [m.get(metric_name, 0) for m in val_metrics]
        
        axes[1].plot(epochs, train_metric_vals, 'b-', label=f'Training {metric_name.title()}', linewidth=2)
        axes[1].plot(epochs, val_metric_vals, 'r-', label=f'Validation {metric_name.title()}', linewidth=2)
        axes[1].set_title(f'Training and Validation {metric_name.title()}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name.title())
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No metrics available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Metrics')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_gan_losses(g_losses: List[float], d_losses: List[float],
                   title: str = "GAN Training Losses",
                   figsize: Tuple[int, int] = (10, 6),
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot GAN generator and discriminator losses.
    
    Args:
        g_losses: Generator losses over epochs
        d_losses: Discriminator losses over epochs
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(g_losses) + 1)
    
    ax.plot(epochs, g_losses, 'b-', label='Generator Loss', linewidth=2)
    ax.plot(epochs, d_losses, 'r-', label='Discriminator Loss', linewidth=2)
    
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                           title: str = "Predictions vs True Values",
                           figsize: Tuple[int, int] = (8, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot predictions vs true values for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot
    
    Returns:
        Matplotlib figure
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def compute_model_complexity(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Compute model complexity metrics.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary containing complexity metrics
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'non_trainable_parameters': total_params - trainable_params
    }