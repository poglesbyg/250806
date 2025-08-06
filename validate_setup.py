#!/usr/bin/env python3
"""
Validation script for PyTorch Showcase project setup.

This script validates that all components are properly installed and functional.
"""

import sys
import importlib
from pathlib import Path

def check_imports():
    """Check that all required packages can be imported."""
    print("🔍 Checking imports...")
    
    required_packages = [
        'torch', 'torchvision', 'torchaudio', 
        'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'tqdm', 'transformers'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def check_project_structure():
    """Check that all project files exist."""
    print("\n🏗️  Checking project structure...")
    
    required_files = [
        'main.py',
        'pyproject.toml', 
        'README.md',
        'pytorch_showcase/__init__.py',
        'pytorch_showcase/models/cnn.py',
        'pytorch_showcase/models/transformer.py',
        'pytorch_showcase/models/gan.py',
        'pytorch_showcase/utils/trainer.py',
        'pytorch_showcase/utils/data_utils.py',
        'pytorch_showcase/utils/metrics.py',
        'pytorch_showcase/utils/visualizer.py',
        'pytorch_showcase/demos/cnn_demo.py',
        'pytorch_showcase/demos/transformer_demo.py',
        'pytorch_showcase/demos/gan_demo.py',
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_models():
    """Check that models can be instantiated."""
    print("\n🧠 Checking models...")
    
    try:
        from pytorch_showcase.models import CIFAR10CNN
        model = CIFAR10CNN()
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ CNN: {param_count:,} parameters")
    except Exception as e:
        print(f"  ❌ CNN: {e}")
        return False
    
    try:
        from pytorch_showcase.models.transformer import create_text_classifier
        model = create_text_classifier(vocab_size=1000, num_classes=3)
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✅ Transformer: {param_count:,} parameters")
    except Exception as e:
        print(f"  ❌ Transformer: {e}")
        return False
    
    try:
        from pytorch_showcase.models import SimpleGAN
        model = SimpleGAN()
        g_params = sum(p.numel() for p in model.generator.parameters())
        d_params = sum(p.numel() for p in model.discriminator.parameters())
        print(f"  ✅ GAN: Generator {g_params:,}, Discriminator {d_params:,} parameters")
    except Exception as e:
        print(f"  ❌ GAN: {e}")
        return False
    
    return True

def check_demos():
    """Check that demo functions can be imported."""
    print("\n🎮 Checking demos...")
    
    try:
        from pytorch_showcase.demos import run_cnn_demo, run_transformer_demo, run_gan_demo
        print("  ✅ All demo functions imported successfully")
        return True
    except Exception as e:
        print(f"  ❌ Demo imports failed: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔥 PyTorch Showcase - Setup Validation")
    print("=" * 50)
    
    checks = [
        ("Package Imports", check_imports),
        ("Project Structure", check_project_structure), 
        ("Model Creation", check_models),
        ("Demo Functions", check_demos),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            passed = check_func()
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"\n❌ {check_name} check failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All validation checks passed!")
        print("🚀 Ready to run: uv run python main.py --help")
        return 0
    else:
        print("❌ Some validation checks failed!")
        print("Please check the errors above and fix any issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())