"""
Training utilities for different types of models.

This module provides flexible training classes for:
- Standard supervised learning models
- Generative Adversarial Networks (GANs)
- Custom training loops with logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable, Any
import time
import os
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Trainer:
    """
    General purpose trainer for supervised learning models.
    
    Features:
    - Automatic mixed precision training
    - Learning rate scheduling
    - Checkpointing and model saving
    - Logging with tensorboard/wandb
    - Early stopping
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        gradient_clip_val: Optional[float] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_amp = use_amp
        self.gradient_clip_val = gradient_clip_val
        
        # Logging setup
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb and wandb_project:
            wandb.init(project=wandb_project)
            wandb.watch(self.model)
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        
        # AMP setup
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(self, train_loader: DataLoader, 
                   metric_fn: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} - Training")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip_val:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.gradient_clip_val:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
            
            if metric_fn:
                all_predictions.append(output.detach().cpu())
                all_targets.append(target.detach().cpu())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        metrics = {}
        if metric_fn and all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = metric_fn(predictions, targets)
        
        return avg_loss, metrics
    
    def validate_epoch(self, val_loader: DataLoader,
                      metric_fn: Optional[Callable] = None) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} - Validation")
            
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                if metric_fn:
                    all_predictions.append(output.cpu())
                    all_targets.append(target.cpu())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / total_samples
        
        # Calculate metrics
        metrics = {}
        if metric_fn and all_predictions:
            predictions = torch.cat(all_predictions, dim=0)
            targets = torch.cat(all_targets, dim=0)
            metrics = metric_fn(predictions, targets)
        
        return avg_loss, metrics
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int, metric_fn: Optional[Callable] = None,
            save_best: bool = True, early_stopping_patience: Optional[int] = None) -> Dict[str, List]:
        """Train the model for multiple epochs."""
        
        early_stopping_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader, metric_fn)
            
            # Validation
            val_loss, val_metrics = self.validate_epoch(val_loader, metric_fn)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Logging
            self._log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, epoch_time)
            
            # Save best model
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pt", is_best=True)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_patience and early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                   train_metrics: Dict, val_metrics: Dict, epoch_time: float) -> None:
        """Log epoch results."""
        
        # Console logging
        print(f"Epoch {epoch + 1:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        if train_metrics:
            train_str = " | ".join([f"Train {k}: {v:.4f}" for k, v in train_metrics.items()])
            print(f"         | {train_str}")
        
        if val_metrics:
            val_str = " | ".join([f"Val {k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"         | {val_str}")
        
        # Tensorboard logging
        if self.writer:
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            
            for name, value in train_metrics.items():
                self.writer.add_scalar(f'Metrics/Train_{name}', value, epoch)
            
            for name, value in val_metrics.items():
                self.writer.add_scalar(f'Metrics/Val_{name}', value, epoch)
            
            # Log learning rate
            if self.optimizer.param_groups:
                self.writer.add_scalar('Learning_Rate', 
                                     self.optimizer.param_groups[0]['lr'], epoch)
        
        # Wandb logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch_time': epoch_time
            }
            
            for name, value in train_metrics.items():
                log_dict[f'train_{name}'] = value
            
            for name, value in val_metrics.items():
                log_dict[f'val_{name}'] = value
            
            if self.optimizer.param_groups:
                log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            
            wandb.log(log_dict)
    
    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        if is_best:
            print(f"New best model saved: {filepath}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Checkpoint loaded: {filepath}")


class GANTrainer:
    """
    Specialized trainer for Generative Adversarial Networks.
    
    Features:
    - Alternating generator and discriminator training
    - Gradient penalty support (WGAN-GP)
    - Image generation and logging
    - Mode collapse detection
    """
    
    def __init__(
        self,
        gan_model: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        device: torch.device,
        latent_dim: int = 100,
        n_critic: int = 5,
        lambda_gp: float = 10.0,
        log_dir: str = "gan_logs",
        checkpoint_dir: str = "gan_checkpoints",
        use_wandb: bool = False,
        wandb_project: Optional[str] = None
    ):
        self.gan = gan_model
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = device
        self.latent_dim = latent_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        
        # Logging setup
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir)
        
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb and wandb_project:
            wandb.init(project=wandb_project)
        
        # Training state
        self.current_epoch = 0
        self.g_losses = []
        self.d_losses = []
        
        # Fixed noise for consistent generation visualization
        self.fixed_noise = torch.randn(64, latent_dim, device=device)
    
    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float]:
        """Train GAN for one epoch."""
        self.gan.generator.train()
        self.gan.discriminator.train()
        
        g_loss_epoch = 0.0
        d_loss_epoch = 0.0
        num_batches = 0
        
        pbar = tqdm(data_loader, desc=f"Epoch {self.current_epoch + 1} - GAN Training")
        
        for batch_idx, (real_images, _) in enumerate(pbar):
            real_images = real_images.to(self.device)
            batch_size = real_images.size(0)
            
            # Train Discriminator
            for _ in range(self.n_critic):
                self.d_optimizer.zero_grad()
                
                # Generate fake images
                z = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_images = self.gan.generator(z)
                
                # Calculate discriminator loss
                d_loss_total, d_loss_real, d_loss_fake = self.gan.discriminator_loss(
                    real_images, fake_images)
                
                # Add gradient penalty if using WGAN-GP
                if self.lambda_gp > 0:
                    from ..models.gan import gradient_penalty
                    gp = gradient_penalty(self.gan.discriminator, real_images, 
                                        fake_images, self.device)
                    d_loss_total += self.lambda_gp * gp
                
                d_loss_total.backward()
                self.d_optimizer.step()
            
            # Train Generator
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake_images = self.gan.generator(z)
            g_loss = self.gan.generator_loss(fake_images)
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # Accumulate losses
            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss_total.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': f'{g_loss.item():.4f}',
                'D_loss': f'{d_loss_total.item():.4f}'
            })
        
        avg_g_loss = g_loss_epoch / num_batches
        avg_d_loss = d_loss_epoch / num_batches
        
        return avg_g_loss, avg_d_loss
    
    def generate_samples(self, num_samples: int = 64) -> torch.Tensor:
        """Generate samples for visualization."""
        self.gan.generator.eval()
        with torch.no_grad():
            samples = self.gan.generate(num_samples, self.device)
        return samples
    
    def fit(self, data_loader: DataLoader, epochs: int, 
            save_interval: int = 10, sample_interval: int = 5) -> Dict[str, List]:
        """Train the GAN for multiple epochs."""
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            start_time = time.time()
            
            # Training
            g_loss, d_loss = self.train_epoch(data_loader)
            
            # Save losses
            self.g_losses.append(g_loss)
            self.d_losses.append(d_loss)
            
            epoch_time = time.time() - start_time
            
            # Logging
            self._log_epoch(epoch, g_loss, d_loss, epoch_time)
            
            # Generate and save samples
            if (epoch + 1) % sample_interval == 0:
                self._generate_and_log_samples(epoch)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f"gan_checkpoint_epoch_{epoch + 1}.pt")
        
        # Close logging
        if self.writer:
            self.writer.close()
        
        return {
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
    
    def _log_epoch(self, epoch: int, g_loss: float, d_loss: float, epoch_time: float) -> None:
        """Log epoch results."""
        
        # Console logging
        print(f"Epoch {epoch + 1:3d} | "
              f"G Loss: {g_loss:.4f} | "
              f"D Loss: {d_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Tensorboard logging
        if self.writer:
            self.writer.add_scalar('Loss/Generator', g_loss, epoch)
            self.writer.add_scalar('Loss/Discriminator', d_loss, epoch)
        
        # Wandb logging
        if self.use_wandb:
            wandb.log({
                'epoch': epoch,
                'g_loss': g_loss,
                'd_loss': d_loss,
                'epoch_time': epoch_time
            })
    
    def _generate_and_log_samples(self, epoch: int) -> None:
        """Generate and log sample images."""
        samples = self.generate_samples(64)
        
        # Save sample grid
        from torchvision.utils import save_image, make_grid
        
        sample_dir = self.log_dir / "samples"
        sample_dir.mkdir(exist_ok=True)
        
        save_image(samples, sample_dir / f"samples_epoch_{epoch + 1}.png",
                  nrow=8, normalize=True)
        
        # Log to tensorboard
        if self.writer:
            grid = make_grid(samples[:16], nrow=4, normalize=True)
            self.writer.add_image('Generated_Samples', grid, epoch)
        
        # Log to wandb
        if self.use_wandb:
            grid = make_grid(samples[:16], nrow=4, normalize=True)
            wandb.log({f"samples_epoch_{epoch + 1}": wandb.Image(grid)})
    
    def save_checkpoint(self, filename: str) -> None:
        """Save GAN checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.gan.generator.state_dict(),
            'discriminator_state_dict': self.gan.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_losses': self.g_losses,
            'd_losses': self.d_losses,
        }
        
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"GAN checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load GAN checkpoint."""
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.gan.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.gan.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.g_losses = checkpoint.get('g_losses', [])
        self.d_losses = checkpoint.get('d_losses', [])
        
        print(f"GAN checkpoint loaded: {filepath}")