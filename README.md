# 🔥 PyTorch Showcase

A comprehensive demonstration of PyTorch capabilities featuring modern deep learning architectures and techniques.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

This project showcases three major areas of deep learning:

### 🖼️ Computer Vision - CNN
- **ResNet-inspired CNN** for CIFAR-10 classification
- Residual connections and batch normalization
- Data augmentation and mixed precision training
- Feature map and filter visualization
- Grad-CAM for explainable AI

### 🤖 Natural Language Processing - Transformer
- **Custom Transformer** architecture for text classification
- Multi-head self-attention mechanism
- Positional encoding and layer normalization
- Attention weight visualization
- Gradient clipping and learning rate scheduling

### 🎨 Generative AI - GAN
- **DCGAN** for MNIST image generation
- Generator and discriminator networks
- Adversarial training with spectral normalization
- Progressive sample generation
- Latent space interpolation

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pytorch-showcase
```

2. Install dependencies using uv:
```bash
uv sync
```

### Running the Demos

#### Run All Demonstrations
```bash
uv run python main.py --all
```

#### Run Individual Demos
```bash
# CNN Demo (CIFAR-10 Classification)
uv run python main.py --cnn

# Transformer Demo (Text Classification)  
uv run python main.py --transformer

# GAN Demo (MNIST Generation)
uv run python main.py --gan
```

#### Custom Parameters
```bash
# Run CNN with custom settings
uv run python main.py --cnn --epochs 30 --batch-size 64 --learning-rate 0.01

# Run Transformer with large model
uv run python main.py --transformer --model-size large --epochs 20

# Run conditional GAN
uv run python main.py --gan --gan-type conditional --epochs 100
```

## 📊 What You'll Get

After running the demos, you'll find comprehensive outputs in the `./outputs/` directory:

### CNN Results
- **Training curves**: Loss and accuracy over epochs
- **Confusion matrix**: Detailed classification performance
- **Feature maps**: Visualization of learned features
- **Filters**: Convolutional filter visualizations
- **Grad-CAM**: Attention heatmaps for explainability

### Transformer Results
- **Training progress**: Loss and metrics tracking
- **Attention weights**: Multi-head attention visualizations
- **Sample predictions**: Model predictions on test examples
- **Confusion matrix**: Classification performance analysis

### GAN Results
- **Generated samples**: High-quality generated images
- **Training dynamics**: Generator vs discriminator loss curves
- **Latent interpolation**: Smooth transitions in latent space
- **Sample diversity**: Quality assessment of generated images
- **Conditional samples** (if using conditional GAN)

## 🏗️ Project Structure

```
pytorch-showcase/
├── pytorch_showcase/
│   ├── models/                 # Neural network architectures
│   │   ├── cnn.py             # ResNet-inspired CNN
│   │   ├── transformer.py     # Custom transformer
│   │   └── gan.py             # DCGAN implementation
│   ├── utils/                 # Utilities and tools
│   │   ├── trainer.py         # Training loops
│   │   ├── data_utils.py      # Data loading
│   │   ├── metrics.py         # Evaluation metrics
│   │   └── visualizer.py      # Visualization tools
│   └── demos/                 # Demo scripts
│       ├── cnn_demo.py
│       ├── transformer_demo.py
│       └── gan_demo.py
├── main.py                    # Main entry point
├── pyproject.toml            # Project dependencies
└── README.md                 # This file
```

## 🔧 Advanced Usage

### Custom Training

You can import and use the models directly:

```python
from pytorch_showcase.models import CIFAR10CNN, TextClassifier, SimpleGAN
from pytorch_showcase.utils import Trainer, get_cifar10_loaders

# Create and train a custom CNN
model = CIFAR10CNN(num_classes=10)
train_loader, val_loader, test_loader = get_cifar10_loaders()

trainer = Trainer(model, criterion, optimizer, device)
results = trainer.fit(train_loader, val_loader, epochs=50)
```

### Jupyter Notebooks

Interactive notebooks are available in the `notebooks/` directory:
- `cnn_exploration.ipynb`: Deep dive into CNN architectures
- `transformer_attention.ipynb`: Attention mechanism analysis
- `gan_latent_space.ipynb`: Exploring GAN latent spaces

### Weights & Biases Integration

Enable experiment tracking with W&B:

```bash
uv run python main.py --all --wandb
```

## 🎯 Key Learning Objectives

This project demonstrates:

1. **Modern PyTorch Patterns**
   - Custom datasets and data loaders
   - Mixed precision training
   - Learning rate scheduling
   - Gradient clipping

2. **Model Architecture Design**
   - Residual connections
   - Multi-head attention
   - Adversarial training
   - Batch/layer normalization

3. **Training Best Practices**
   - Data augmentation
   - Early stopping
   - Checkpointing
   - Hyperparameter tuning

4. **Evaluation and Visualization**
   - Comprehensive metrics
   - Feature visualization
   - Attention analysis
   - Generated sample quality

## 🧪 Experimental Features

### Model Variants
- **CNN**: Different depths and widths
- **Transformer**: Small, base, and large configurations
- **GAN**: Simple and conditional variants

### Advanced Techniques
- Spectral normalization for GANs
- Gradient penalty (WGAN-GP)
- MixUp and CutMix augmentation
- OneCycle learning rate scheduling

## 📈 Performance Benchmarks

Typical results on default settings:

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| CNN | CIFAR-10 | Accuracy | ~85-90% |
| Transformer | Text Classification | Accuracy | ~80-85% |
| GAN | MNIST Generation | FID | ~20-30 |

*Results may vary based on hardware and random initialization*

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional model architectures (Vision Transformer, BERT, StyleGAN)
- More datasets and benchmarks
- Advanced training techniques
- Mobile/edge deployment examples

## 📚 Educational Resources

This project serves as a practical companion to:
- Deep Learning courses
- PyTorch tutorials
- Computer vision and NLP research
- Machine learning engineering practices

## 🐛 Troubleshooting

### Common Issues

**CUDA out of memory**:
```bash
# Reduce batch size
uv run python main.py --cnn --batch-size 32

# Use CPU
uv run python main.py --all --device cpu
```

**Slow training**:
- Ensure CUDA is available and properly installed
- Consider using smaller model sizes
- Reduce number of epochs for quick testing

**Import errors**:
```bash
# Reinstall dependencies
uv sync --force
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent framework
- Research papers that inspired the architectures
- Open source community for tools and libraries

---

**Happy Learning! 🎓**

*This project is designed for educational purposes to showcase PyTorch capabilities and modern deep learning techniques.*