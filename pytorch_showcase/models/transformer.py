"""
Transformer model for text classification.

This module implements a modern transformer architecture with:
- Multi-head self-attention
- Position encoding
- Layer normalization
- Feed-forward networks
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""
    
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through multi-head attention."""
        batch_size = query.size(0)
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        output = self.w_o(context)
        
        return output, attention_weights


class TransformerBlock(nn.Module):
    """A single transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class TextClassifier(nn.Module):
    """
    Transformer-based text classifier.
    
    Architecture:
    - Token embedding layer
    - Positional encoding
    - Stack of transformer blocks
    - Global average pooling
    - Classification head
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0, 0.1)
    
    def create_padding_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create mask to ignore padding tokens."""
        return (x != self.pad_token_id).unsqueeze(1).unsqueeze(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer."""
        # Create padding mask
        mask = self.create_padding_mask(x)
        
        # Token embeddings with scaling
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Pass through transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        
        # Normalize and pool
        x = self.norm(x)
        
        # Global average pooling (ignoring padding tokens)
        mask_expanded = mask.squeeze(1).squeeze(1).unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        # Classification
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """Get attention weights from a specific layer for visualization."""
        mask = self.create_padding_mask(x)
        
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)
        
        for i, transformer in enumerate(self.transformer_blocks):
            if i == layer_idx or (layer_idx == -1 and i == len(self.transformer_blocks) - 1):
                _, attention_weights = transformer.attention(x, x, x, mask)
                return attention_weights
            x = transformer(x, mask)
        
        return None


def create_text_classifier(
    vocab_size: int,
    num_classes: int,
    model_size: str = "base",
    **kwargs
) -> TextClassifier:
    """Create a text classifier with predefined sizes."""
    
    size_configs = {
        "small": {"d_model": 256, "num_heads": 4, "num_layers": 4, "d_ff": 1024},
        "base": {"d_model": 512, "num_heads": 8, "num_layers": 6, "d_ff": 2048},
        "large": {"d_model": 768, "num_heads": 12, "num_layers": 12, "d_ff": 3072},
    }
    
    config = size_configs.get(model_size, size_configs["base"])
    config.update(kwargs)
    
    return TextClassifier(vocab_size=vocab_size, num_classes=num_classes, **config)