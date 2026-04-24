"""
Lightweight Streaming Transformer for LSST Alert Processing.
Optimized for low-latency inference on streaming time-series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Tuple, Optional


class StreamingAttention(nn.Module):
    """Efficient attention mechanism with fixed-size context window for streaming."""
    
    def __init__(self, d_model: int, n_heads: int = 4, window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.scale = self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # KV cache for streaming (batch size will be set dynamically)
        self.kv_cache_k = None
        self.kv_cache_v = None
        self.cache_pos = 0
        self.current_batch_size = 0
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [B, 1, D] - streaming query (latest token)
            k: [B, 1, D] - streaming key (latest token)
            v: [B, 1, D] - streaming value (latest token)
        Returns:
            output: [B, 1, D]
            attn_weights: [B, n_heads, 1, window_size]
        """
        B = q.size(0)
        
        # Initialize cache if needed or batch size changed
        if self.kv_cache_k is None or self.current_batch_size != B:
            self.kv_cache_k = torch.zeros(B, self.window_size, self.n_heads, self.head_dim, 
                                         device=q.device, dtype=q.dtype)
            self.kv_cache_v = torch.zeros(B, self.window_size, self.n_heads, self.head_dim, 
                                         device=q.device, dtype=q.dtype)
            self.current_batch_size = B
            self.cache_pos = 0
        
        # Project to multiple heads
        Q = self.W_q(q).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, 1, head_dim]
        K = self.W_k(k).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Update KV cache (circular buffer)
        cache_idx = self.cache_pos % self.window_size
        self.kv_cache_k[:, cache_idx:cache_idx+1] = K.transpose(1, 2)  # [B, 1, n_heads, head_dim]
        self.kv_cache_v[:, cache_idx:cache_idx+1] = V.transpose(1, 2)
        self.cache_pos += 1
        
        # Reshape cache for attention: [B, window_size, n_heads, head_dim] -> [B, n_heads, window_size, head_dim]
        K_cached = self.kv_cache_k.transpose(1, 2)  # [B, n_heads, window_size, head_dim]
        V_cached = self.kv_cache_v.transpose(1, 2)
        
        # Compute attention: Q is [B, n_heads, 1, head_dim], K_cached is [B, n_heads, window_size, head_dim]
        scores = torch.matmul(Q, K_cached.transpose(-1, -2)) * self.scale  # [B, n_heads, 1, window_size]
        
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V_cached)  # [B, n_heads, 1, head_dim]
        output = output.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights
    
    def reset_cache(self):
        """Reset KV cache for new sequence."""
        self.cache_pos = 0
        if self.kv_cache_k is not None:
            self.kv_cache_k.zero_()
        if self.kv_cache_v is not None:
            self.kv_cache_v.zero_()
        self.current_batch_size = 0


class StreamingTransformerLayer(nn.Module):
    """Single streaming transformer layer optimized for online processing."""
    
    def __init__(self, d_model: int = 64, n_heads: int = 4, d_ff: int = 256, 
                 window_size: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.attention = StreamingAttention(d_model, n_heads, window_size, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Lightweight FFN with SwiGLU
        self.ffn_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_up = nn.Linear(d_model, d_ff, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, D] - streaming input
        Returns:
            output: [B, 1, D]
        """
        # Self-attention with pre-norm
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # SwiGLU FFN with pre-norm
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        ffn_out = self.ffn_down(gate * up)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        
        return x


class StreamingTransformer(nn.Module):
    """Lightweight streaming transformer for real-time LSST alert processing."""
    
    def __init__(self, input_dim: int, d_model: int = 64, n_layers: int = 2, 
                 n_heads: int = 4, d_ff: int = 256, window_size: int = 32, 
                 output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model, bias=False)
        
        self.layers = nn.ModuleList([
            StreamingTransformerLayer(d_model, n_heads, d_ff, window_size, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, output_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, input_dim] or [B, input_dim]
        Returns:
            embeddings: [B, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, input_dim]
        
        x = self.input_projection(x)  # [B, 1, d_model]
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.output_projection(x)  # [B, 1, output_dim]
        return x.squeeze(1)  # [B, output_dim]
    
    def reset_cache(self):
        """Reset attention caches for new sequence."""
        for layer in self.layers:
            layer.attention.reset_cache()
