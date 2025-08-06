"""
PyTorch Showcase - Main Demo Script

This script demonstrates various PyTorch capabilities including:
- Convolutional Neural Networks for image classification
- Transformers for text classification  
- Generative Adversarial Networks for image generation

Run different demos with command line arguments or run all demos sequentially.
"""

import argparse
import sys
from pathlib import Path
import torch
import time

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pytorch_showcase.demos import run_cnn_demo, run_transformer_demo, run_gan_demo


def main():
    """Main function to run PyTorch demonstrations."""
    
    parser = argparse.ArgumentParser(
        description="PyTorch Showcase - Demonstrate various deep learning techniques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all                    # Run all demos
  python main.py --cnn                    # Run only CNN demo
  python main.py --transformer            # Run only Transformer demo
  python main.py --gan                    # Run only GAN demo
  python main.py --cnn --epochs 10        # Run CNN with custom epochs
  python main.py --all --device cpu       # Run all demos on CPU
        """
    )
    
    # Demo selection
    parser.add_argument('--all', action='store_true', help='Run all demonstrations')
    parser.add_argument('--cnn', action='store_true', help='Run CNN demonstration')
    parser.add_argument('--transformer', action='store_true', help='Run Transformer demonstration')
    parser.add_argument('--gan', action='store_true', help='Run GAN demonstration')
    
    # Common parameters
    parser.add_argument('--epochs', type=int, default=None, 
                       help='Number of training epochs (default varies by model)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training (default varies by model)')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate (default varies by model)')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training (default: auto)')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Directory for data storage (default: ./data)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory for outputs (default: ./outputs)')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    
    # Model-specific parameters
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='Transformer model size (default: small)')
    parser.add_argument('--gan-type', type=str, default='simple',
                       choices=['simple', 'conditional'],
                       help='Type of GAN to use (default: simple)')
    parser.add_argument('--latent-dim', type=int, default=100,
                       help='Latent dimension for GAN (default: 100)')
    
    args = parser.parse_args()
    
    # If no specific demo is selected, show help
    if not (args.all or args.cnn or args.transformer or args.gan):
        parser.print_help()
        return
    
    # Print system info
    print("🔥 PyTorch Showcase")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    total_start_time = time.time()
    
    # Run CNN demo
    if args.all or args.cnn:
        print("\n" + "="*80)
        print("RUNNING CNN DEMONSTRATION")
        print("="*80)
        
        try:
            cnn_results = run_cnn_demo(
                epochs=args.epochs or 20,
                batch_size=args.batch_size or 128,
                learning_rate=args.learning_rate or 0.001,
                device=args.device,
                data_dir=args.data_dir,
                output_dir=f"{args.output_dir}/cnn",
                use_wandb=args.wandb
            )
            results['cnn'] = cnn_results
            print(f"✅ CNN demo completed successfully!")
            
        except Exception as e:
            print(f"❌ CNN demo failed: {str(e)}")
            if not args.all:  # If running only CNN, exit on failure
                return
    
    # Run Transformer demo
    if args.all or args.transformer:
        print("\n" + "="*80)
        print("RUNNING TRANSFORMER DEMONSTRATION")
        print("="*80)
        
        try:
            transformer_results = run_transformer_demo(
                epochs=args.epochs or 15,
                batch_size=args.batch_size or 32,
                learning_rate=args.learning_rate or 1e-4,
                device=args.device,
                output_dir=f"{args.output_dir}/transformer",
                use_wandb=args.wandb,
                model_size=args.model_size
            )
            results['transformer'] = transformer_results
            print(f"✅ Transformer demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Transformer demo failed: {str(e)}")
            if not args.all:  # If running only Transformer, exit on failure
                return
    
    # Run GAN demo
    if args.all or args.gan:
        print("\n" + "="*80)
        print("RUNNING GAN DEMONSTRATION")
        print("="*80)
        
        try:
            gan_results = run_gan_demo(
                epochs=args.epochs or 50,
                batch_size=args.batch_size or 64,
                g_lr=args.learning_rate or 2e-4,
                d_lr=args.learning_rate or 2e-4,
                latent_dim=args.latent_dim,
                device=args.device,
                data_dir=args.data_dir,
                output_dir=f"{args.output_dir}/gan",
                use_wandb=args.wandb,
                gan_type=args.gan_type
            )
            results['gan'] = gan_results
            print(f"✅ GAN demo completed successfully!")
            
        except Exception as e:
            print(f"❌ GAN demo failed: {str(e)}")
            if not args.all:  # If running only GAN, exit on failure
                return
    
    # Print final summary
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"Total execution time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {args.output_dir}")
    
    if 'cnn' in results:
        cnn_acc = results['cnn']['test_metrics'].get('accuracy', 0)
        print(f"CNN Test Accuracy: {cnn_acc:.3f}")
    
    if 'transformer' in results:
        transformer_acc = results['transformer']['test_metrics'].get('accuracy', 0)
        print(f"Transformer Test Accuracy: {transformer_acc:.3f}")
    
    if 'gan' in results:
        final_g_loss = results['gan']['training_results']['g_losses'][-1]
        final_d_loss = results['gan']['training_results']['d_losses'][-1]
        print(f"GAN Final Losses - Generator: {final_g_loss:.3f}, Discriminator: {final_d_loss:.3f}")
    
    print("\n🎉 All demonstrations completed!")
    print("\nGenerated outputs:")
    print(f"  - Training curves and metrics plots")
    print(f"  - Model visualizations and feature maps")
    print(f"  - Generated samples (for GAN)")
    print(f"  - Attention visualizations (for Transformer)")
    print(f"  - Confusion matrices and evaluation results")
    
    print(f"\nExplore the results in: {args.output_dir}")


if __name__ == "__main__":
    main()