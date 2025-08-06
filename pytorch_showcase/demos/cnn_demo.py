"""
CNN Demo: CIFAR-10 Image Classification

This demo showcases:
- CNN architecture with residual connections
- Data loading and augmentation
- Training with modern techniques (mixed precision, scheduling)
- Model evaluation and visualization
- Feature map and filter visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from pathlib import Path

from ..models.cnn import CIFAR10CNN
from ..utils.data_utils import get_cifar10_loaders, denormalize_cifar10, get_class_names
from ..utils.trainer import Trainer
from ..utils.metrics import compute_metrics, plot_confusion_matrix, plot_training_curves
from ..utils.visualizer import ModelVisualizer, visualize_data_samples


def run_cnn_demo(
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    device: str = "auto",
    data_dir: str = "./data",
    output_dir: str = "./outputs/cnn",
    use_wandb: bool = False
) -> dict:
    """
    Run CNN demonstration on CIFAR-10.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to use ('auto', 'cpu', 'cuda')
        data_dir: Directory for data storage
        output_dir: Directory for outputs
        use_wandb: Whether to use Weights & Biases logging
    
    Returns:
        Dictionary with training results and model
    """
    
    print("🔥 Starting CNN Demo: CIFAR-10 Image Classification")
    print("=" * 60)
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📊 Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        val_split=0.1,
        augment_train=True
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some data samples
    print("\n🖼️  Visualizing data samples...")
    class_names = get_class_names("cifar10")
    
    fig = visualize_data_samples(
        train_loader, 
        class_names=class_names,
        num_samples=16,
        denormalize_fn=denormalize_cifar10,
        save_path=output_path / "data_samples.png"
    )
    plt.close(fig)
    
    # Create model
    print("\n🏗️  Creating CNN model...")
    model = CIFAR10CNN(num_classes=10, dropout_rate=0.3)
    model = model.to(device)
    
    # Print model summary
    from ..utils.visualizer import create_model_summary
    summary = create_model_summary(model, (3, 32, 32), device)
    print(summary)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=True,
        gradient_clip_val=1.0,
        log_dir=str(output_path / "logs"),
        checkpoint_dir=str(output_path / "checkpoints"),
        use_wandb=use_wandb,
        wandb_project="pytorch-showcase-cnn"
    )
    
    # Define metrics function
    def compute_classification_metrics(predictions, targets):
        return compute_metrics(predictions, targets, task_type="classification")
    
    # Train model
    print(f"\n🚀 Training CNN for {epochs} epochs...")
    training_results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        metric_fn=compute_classification_metrics,
        save_best=True,
        early_stopping_patience=10
    )
    
    # Plot training curves
    print("\n📈 Plotting training curves...")
    fig = plot_training_curves(
        training_results['train_losses'],
        training_results['val_losses'],
        training_results['train_metrics'],
        training_results['val_metrics'],
        metric_name='accuracy',
        title="CNN Training Progress",
        save_path=output_path / "training_curves.png"
    )
    plt.close(fig)
    
    # Load best model for evaluation
    print("\n🔍 Evaluating best model...")
    trainer.load_checkpoint("best_model.pt")
    
    # Test evaluation
    model.eval()
    all_predictions = []
    all_targets = []
    test_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    # Compute test metrics
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    test_metrics = compute_classification_metrics(predictions, targets)
    
    print(f"\n📊 Test Results:")
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
    for metric, value in test_metrics.items():
        if not metric.startswith(('precision_class', 'recall_class', 'f1_class')):
            print(f"{metric.title()}: {value:.4f}")
    
    # Plot confusion matrix
    pred_classes = predictions.argmax(dim=1).numpy()
    true_classes = targets.numpy()
    
    fig = plot_confusion_matrix(
        true_classes, pred_classes,
        class_names=class_names,
        normalize=True,
        title="CNN Confusion Matrix (Normalized)",
        save_path=output_path / "confusion_matrix.png"
    )
    plt.close(fig)
    
    # Visualize model internals
    print("\n🔬 Visualizing model internals...")
    visualizer = ModelVisualizer(model, device)
    
    # Get a sample for visualization
    sample_data, _ = next(iter(test_loader))
    sample_input = sample_data[:1].to(device)
    
    # Visualize feature maps
    try:
        fig = visualizer.visualize_feature_maps(
            sample_input, 
            layer_name="layer1.0.conv1",
            num_features=16,
            save_path=output_path / "feature_maps.png"
        )
        plt.close(fig)
        
        # Visualize filters
        fig = visualizer.visualize_filters(
            layer_name="conv1",
            num_filters=16,
            save_path=output_path / "filters.png"
        )
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not visualize internals: {e}")
    
    # Grad-CAM visualization
    try:
        sample_class = targets[0].item()
        fig = visualizer.visualize_grad_cam(
            sample_input,
            target_class=sample_class,
            target_layer="layer3.1.conv2",
            save_path=output_path / "grad_cam.png"
        )
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not create Grad-CAM: {e}")
    
    print(f"\n✅ CNN Demo completed! Results saved to: {output_path}")
    
    return {
        'model': model,
        'training_results': training_results,
        'test_metrics': test_metrics,
        'class_names': class_names,
        'device': device
    }