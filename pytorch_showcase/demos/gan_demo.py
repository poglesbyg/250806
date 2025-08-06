"""
GAN Demo: Image Generation

This demo showcases:
- DCGAN architecture with generator and discriminator
- Adversarial training with alternating optimization
- Progressive sample generation during training
- Loss curve analysis and mode collapse detection
- Generated image visualization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from torchvision.utils import save_image, make_grid

from ..models.gan import SimpleGAN, create_gan
from ..utils.data_utils import get_mnist_loaders
from ..utils.trainer import GANTrainer
from ..utils.metrics import plot_gan_losses
from ..utils.visualizer import visualize_generated_samples


def run_gan_demo(
    epochs: int = 50,
    batch_size: int = 64,
    g_lr: float = 2e-4,
    d_lr: float = 2e-4,
    latent_dim: int = 100,
    device: str = "auto",
    data_dir: str = "./data",
    output_dir: str = "./outputs/gan",
    use_wandb: bool = False,
    gan_type: str = "simple"
) -> dict:
    """
    Run GAN demonstration on MNIST image generation.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        g_lr: Generator learning rate
        d_lr: Discriminator learning rate
        latent_dim: Latent space dimension
        device: Device to use ('auto', 'cpu', 'cuda')
        data_dir: Directory for data storage
        output_dir: Directory for outputs
        use_wandb: Whether to use Weights & Biases logging
        gan_type: Type of GAN ('simple', 'conditional')
    
    Returns:
        Dictionary with training results and model
    """
    
    print("🎨 Starting GAN Demo: MNIST Image Generation")
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
    print("\n📊 Loading MNIST dataset...")
    train_loader, val_loader, test_loader = get_mnist_loaders(
        batch_size=batch_size,
        data_dir=data_dir,
        val_split=0.0  # No validation needed for GAN
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize real data samples
    print("\n🖼️  Visualizing real data samples...")
    real_batch = next(iter(train_loader))[0]
    fig = visualize_generated_samples(
        real_batch,
        num_samples=16,
        title="Real MNIST Samples",
        save_path=output_path / "real_samples.png"
    )
    plt.close(fig)
    
    # Create GAN model
    print(f"\n🏗️  Creating {gan_type.upper()} GAN model...")
    if gan_type == "conditional":
        gan_model = create_gan(
            gan_type="conditional",
            latent_dim=latent_dim,
            num_classes=10,
            num_channels=1,  # Grayscale MNIST
            feature_map_size=64
        )
    else:
        gan_model = create_gan(
            gan_type="simple",
            latent_dim=latent_dim,
            num_channels=1,  # Grayscale MNIST
            feature_map_size=64,
            use_spectral_norm=True
        )
    
    gan_model = gan_model.to(device)
    
    # Print model summaries
    print("\nGenerator Architecture:")
    from ..utils.visualizer import create_model_summary
    gen_summary = create_model_summary(gan_model.generator, (latent_dim,), device)
    print(gen_summary)
    
    print("\nDiscriminator Architecture:")
    disc_summary = create_model_summary(gan_model.discriminator, (1, 64, 64), device)
    print(disc_summary)
    
    # Setup optimizers
    g_optimizer = optim.Adam(
        gan_model.get_generator_parameters(),
        lr=g_lr,
        betas=(0.5, 0.999)
    )
    
    d_optimizer = optim.Adam(
        gan_model.get_discriminator_parameters(),
        lr=d_lr,
        betas=(0.5, 0.999)
    )
    
    # Create GAN trainer
    trainer = GANTrainer(
        gan_model=gan_model,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        latent_dim=latent_dim,
        n_critic=1,  # Train discriminator once per generator update
        lambda_gp=10.0,  # Gradient penalty weight
        log_dir=str(output_path / "logs"),
        checkpoint_dir=str(output_path / "checkpoints"),
        use_wandb=use_wandb,
        wandb_project="pytorch-showcase-gan"
    )
    
    # Generate initial samples (before training)
    print("\n🎲 Generating initial samples (before training)...")
    initial_samples = trainer.generate_samples(16)
    fig = visualize_generated_samples(
        initial_samples,
        num_samples=16,
        title="Generated Samples (Before Training)",
        save_path=output_path / "initial_samples.png"
    )
    plt.close(fig)
    
    # Train GAN
    print(f"\n🚀 Training GAN for {epochs} epochs...")
    training_results = trainer.fit(
        data_loader=train_loader,
        epochs=epochs,
        save_interval=10,
        sample_interval=5
    )
    
    # Plot training losses
    print("\n📈 Plotting training losses...")
    fig = plot_gan_losses(
        training_results['g_losses'],
        training_results['d_losses'],
        title="GAN Training Losses",
        save_path=output_path / "training_losses.png"
    )
    plt.close(fig)
    
    # Generate final samples
    print("\n🎨 Generating final samples...")
    final_samples = trainer.generate_samples(64)
    
    # Save high-quality grid
    sample_grid = make_grid(final_samples[:64], nrow=8, normalize=True, padding=2)
    save_image(sample_grid, output_path / "final_samples_grid.png")
    
    # Visualize progression
    fig = visualize_generated_samples(
        final_samples,
        num_samples=16,
        title="Generated Samples (After Training)",
        save_path=output_path / "final_samples.png"
    )
    plt.close(fig)
    
    # Generate interpolation samples
    print("\n🌈 Creating latent space interpolation...")
    create_interpolation_samples(
        gan_model.generator, 
        device, 
        latent_dim, 
        output_path / "interpolation.png"
    )
    
    # Analyze training dynamics
    print("\n📊 Analyzing training dynamics...")
    analyze_training_dynamics(
        training_results['g_losses'],
        training_results['d_losses'],
        output_path
    )
    
    # Generate diverse samples for quality assessment
    print("\n🔍 Generating diverse samples for quality assessment...")
    diverse_samples = trainer.generate_samples(100)
    
    # Save samples in grid format
    sample_grids = []
    for i in range(0, 100, 25):
        grid = make_grid(diverse_samples[i:i+25], nrow=5, normalize=True, padding=2)
        sample_grids.append(grid)
    
    # Create quality assessment visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for i, grid in enumerate(sample_grids):
        row, col = i // 2, i % 2
        axes[row, col].imshow(grid.permute(1, 2, 0))
        axes[row, col].set_title(f'Samples {i*25+1}-{(i+1)*25}')
        axes[row, col].axis('off')
    
    plt.suptitle('Generated Sample Diversity Assessment', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "sample_diversity.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Conditional generation if applicable
    if gan_type == "conditional":
        print("\n🏷️  Generating class-conditional samples...")
        create_conditional_samples(gan_model, device, latent_dim, output_path)
    
    print(f"\n✅ GAN Demo completed! Results saved to: {output_path}")
    
    return {
        'model': gan_model,
        'training_results': training_results,
        'final_samples': final_samples,
        'device': device
    }


def create_interpolation_samples(generator: nn.Module, device: torch.device, 
                               latent_dim: int, save_path: Path, num_steps: int = 10) -> None:
    """Create interpolation between two random points in latent space."""
    
    generator.eval()
    
    # Generate two random points
    z1 = torch.randn(1, latent_dim, device=device)
    z2 = torch.randn(1, latent_dim, device=device)
    
    # Create interpolation
    alphas = torch.linspace(0, 1, num_steps).to(device)
    interpolated_samples = []
    
    with torch.no_grad():
        for alpha in alphas:
            z_interp = alpha * z2 + (1 - alpha) * z1
            sample = generator(z_interp)
            interpolated_samples.append(sample)
    
    # Create grid
    interpolated_tensor = torch.cat(interpolated_samples, dim=0)
    grid = make_grid(interpolated_tensor, nrow=num_steps, normalize=True, padding=2)
    save_image(grid, save_path)


def analyze_training_dynamics(g_losses: list, d_losses: list, output_path: Path) -> None:
    """Analyze GAN training dynamics and detect potential issues."""
    
    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(g_losses, label='Generator', alpha=0.7)
    axes[0, 0].plot(d_losses, label='Discriminator', alpha=0.7)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss difference (indicator of balance)
    loss_diff = g_losses - d_losses
    axes[0, 1].plot(loss_diff, color='purple', alpha=0.7)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Loss Difference (G - D)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Difference')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Moving averages (smoothed trends)
    window = min(10, len(g_losses) // 4)
    if window > 1:
        g_smooth = np.convolve(g_losses, np.ones(window)/window, mode='valid')
        d_smooth = np.convolve(d_losses, np.ones(window)/window, mode='valid')
        
        axes[1, 0].plot(g_smooth, label='Generator (smoothed)', linewidth=2)
        axes[1, 0].plot(d_smooth, label='Discriminator (smoothed)', linewidth=2)
        axes[1, 0].set_title('Smoothed Loss Trends')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Loss stability analysis
    g_stability = np.std(g_losses[-20:]) if len(g_losses) > 20 else np.std(g_losses)
    d_stability = np.std(d_losses[-20:]) if len(d_losses) > 20 else np.std(d_losses)
    
    axes[1, 1].bar(['Generator', 'Discriminator'], [g_stability, d_stability], 
                   color=['blue', 'orange'], alpha=0.7)
    axes[1, 1].set_title('Loss Stability (Recent Std Dev)')
    axes[1, 1].set_ylabel('Standard Deviation')
    
    plt.suptitle('GAN Training Dynamics Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "training_analysis.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Print analysis summary
    print("Training Analysis Summary:")
    print(f"  Final Generator Loss: {g_losses[-1]:.4f}")
    print(f"  Final Discriminator Loss: {d_losses[-1]:.4f}")
    print(f"  Generator Stability: {g_stability:.4f}")
    print(f"  Discriminator Stability: {d_stability:.4f}")
    
    # Check for potential issues
    if g_losses[-1] > 5.0:
        print("  ⚠️  Warning: High generator loss might indicate training difficulties")
    
    if d_losses[-1] < 0.1:
        print("  ⚠️  Warning: Very low discriminator loss might indicate overfitting")
    
    if abs(np.mean(loss_diff[-10:])) > 2.0:
        print("  ⚠️  Warning: Large loss imbalance detected")


def create_conditional_samples(gan_model, device: torch.device, 
                             latent_dim: int, output_path: Path) -> None:
    """Create samples for each class in conditional GAN."""
    
    if not hasattr(gan_model, 'generate'):
        return
    
    gan_model.generator.eval()
    
    # Generate samples for each digit (0-9)
    samples_per_class = 8
    all_samples = []
    
    with torch.no_grad():
        for digit in range(10):
            labels = torch.full((samples_per_class,), digit, dtype=torch.long, device=device)
            samples = gan_model.generate(samples_per_class, labels, device)
            all_samples.append(samples)
    
    # Arrange samples by class
    class_samples = torch.cat(all_samples, dim=0)
    
    # Create grid organized by class
    grid = make_grid(class_samples, nrow=samples_per_class, normalize=True, padding=2)
    save_image(grid, output_path / "conditional_samples.png")
    
    # Create individual class visualizations
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for digit in range(10):
        row, col = digit // 5, digit % 5
        
        # Create mini-grid for this digit
        digit_samples = all_samples[digit]
        mini_grid = make_grid(digit_samples, nrow=4, normalize=True, padding=1)
        
        axes[row, col].imshow(mini_grid.permute(1, 2, 0))
        axes[row, col].set_title(f'Digit {digit}')
        axes[row, col].axis('off')
    
    plt.suptitle('Class-Conditional Generated Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path / "conditional_by_class.png", dpi=300, bbox_inches='tight')
    plt.close(fig)