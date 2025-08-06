"""
Transformer Demo: Text Classification

This demo showcases:
- Transformer architecture with multi-head attention
- Text preprocessing and tokenization
- Position encoding and attention mechanisms
- Training with gradient clipping and scheduling
- Attention weight visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from ..models.transformer import TextClassifier
from ..utils.data_utils import get_text_loaders, create_sample_text_data
from ..utils.trainer import Trainer
from ..utils.metrics import compute_metrics, plot_confusion_matrix, plot_training_curves
from ..utils.visualizer import ModelVisualizer


def run_transformer_demo(
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "auto",
    output_dir: str = "./outputs/transformer",
    use_wandb: bool = False,
    model_size: str = "small"
) -> dict:
    """
    Run Transformer demonstration on text classification.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: Device to use ('auto', 'cpu', 'cuda')
        output_dir: Directory for outputs
        use_wandb: Whether to use Weights & Biases logging
        model_size: Model size ('small', 'base', 'large')
    
    Returns:
        Dictionary with training results and model
    """
    
    print("🤖 Starting Transformer Demo: Text Classification")
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
    
    # Create sample text data
    print("\n📝 Creating sample text data...")
    texts, labels = create_sample_text_data(num_samples=2000, num_classes=3)
    
    print(f"Total samples: {len(texts)}")
    print(f"Number of classes: {len(set(labels))}")
    print(f"Sample texts:")
    for i in range(3):
        print(f"  Class {labels[i]}: '{texts[i]}'")
    
    # Load data
    print("\n📊 Creating data loaders...")
    train_loader, val_loader, test_loader, vocab = get_text_loaders(
        texts=texts,
        labels=labels,
        batch_size=batch_size,
        max_length=64,
        val_split=0.2,
        test_split=0.1,
        min_freq=1
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\n🏗️  Creating Transformer model (size: {model_size})...")
    model = TextClassifier(
        vocab_size=len(vocab),
        num_classes=3,
        model_size=model_size,
        max_seq_length=64,
        dropout=0.1
    )
    model = model.to(device)
    
    # Print model summary
    from ..utils.visualizer import create_model_summary
    summary = create_model_summary(model, (64,), device)
    print(summary)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Use OneCycleLR scheduler
    total_steps = len(train_loader) * epochs
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=learning_rate * 10,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=False,  # Often better for transformers to not use AMP
        gradient_clip_val=1.0,
        log_dir=str(output_path / "logs"),
        checkpoint_dir=str(output_path / "checkpoints"),
        use_wandb=use_wandb,
        wandb_project="pytorch-showcase-transformer"
    )
    
    # Define metrics function
    def compute_classification_metrics(predictions, targets):
        return compute_metrics(predictions, targets, task_type="classification")
    
    # Train model
    print(f"\n🚀 Training Transformer for {epochs} epochs...")
    training_results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        metric_fn=compute_classification_metrics,
        save_best=True,
        early_stopping_patience=8
    )
    
    # Plot training curves
    print("\n📈 Plotting training curves...")
    fig = plot_training_curves(
        training_results['train_losses'],
        training_results['val_losses'],
        training_results['train_metrics'],
        training_results['val_metrics'],
        metric_name='accuracy',
        title="Transformer Training Progress",
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
    class_names = ['Negative', 'Neutral', 'Positive']
    pred_classes = predictions.argmax(dim=1).numpy()
    true_classes = targets.numpy()
    
    fig = plot_confusion_matrix(
        true_classes, pred_classes,
        class_names=class_names,
        normalize=True,
        title="Transformer Confusion Matrix (Normalized)",
        save_path=output_path / "confusion_matrix.png"
    )
    plt.close(fig)
    
    # Visualize attention weights
    print("\n🔍 Visualizing attention mechanisms...")
    try:
        # Get a sample for visualization
        sample_data, sample_target = next(iter(test_loader))
        sample_input = sample_data[:1].to(device)
        
        # Get attention weights
        model.eval()
        with torch.no_grad():
            attention_weights = model.get_attention_weights(sample_input, layer_idx=-1)
        
        if attention_weights is not None:
            # Create token list for visualization
            sample_tokens = sample_input[0].cpu().numpy()
            
            # Reverse vocabulary for token lookup
            idx_to_token = {idx: token for token, idx in vocab.items()}
            tokens = [idx_to_token.get(idx, '<UNK>') for idx in sample_tokens[:20]]  # Show first 20 tokens
            
            visualizer = ModelVisualizer(model, device)
            fig = visualizer.visualize_attention_weights(
                sample_input,
                attention_weights[:, :, :20, :20],  # Show 20x20 attention matrix
                input_tokens=tokens,
                save_path=output_path / "attention_weights.png"
            )
            plt.close(fig)
            
    except Exception as e:
        print(f"Warning: Could not visualize attention: {e}")
    
    # Analyze model predictions on sample texts
    print("\n🔍 Analyzing sample predictions...")
    model.eval()
    
    sample_texts = [
        "this movie is amazing and wonderful",
        "this movie is terrible and boring", 
        "this movie is okay and average"
    ]
    
    print("\nSample Predictions:")
    for i, text in enumerate(sample_texts):
        # Tokenize text
        tokens = text.lower().split()
        token_ids = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Pad to model's max length
        if len(token_ids) < 64:
            token_ids.extend([vocab['<PAD>']] * (64 - len(token_ids)))
        else:
            token_ids = token_ids[:64]
        
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
        
        print(f"Text: '{text}'")
        print(f"Predicted: {class_names[predicted_class]} (confidence: {probabilities[0][predicted_class]:.3f})")
        print(f"Probabilities: {[f'{class_names[j]}: {probabilities[0][j]:.3f}' for j in range(3)]}")
        print()
    
    print(f"\n✅ Transformer Demo completed! Results saved to: {output_path}")
    
    return {
        'model': model,
        'training_results': training_results,
        'test_metrics': test_metrics,
        'vocab': vocab,
        'class_names': class_names,
        'device': device
    }