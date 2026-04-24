"""
Online Autoencoder for streaming anomaly detection in LSST alerts.
Uses exponential moving average for parameter updates (no batch processing required).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class StreamingAutoencoder(nn.Module):
    """Lightweight autoencoder optimized for streaming reconstruction and anomaly detection."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 32, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: input_dim -> hidden_dim -> latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )
        
        # Decoder: latent_dim -> hidden_dim -> input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim, bias=False),
        )
        
        # Moving average for online learning
        self.register_buffer('ema_loss', torch.tensor(0.0))
        self.register_buffer('ema_variance', torch.tensor(1.0))
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_dim] or [1, input_dim]
        Returns:
            reconstruction: [B, input_dim]
            latent: [B, latent_dim]
            loss: scalar
        """
        z = self.encode(x)
        recon = self.decode(z)
        
        # Reconstruction loss (MSE)
        loss = F.mse_loss(recon, x, reduction='mean')
        
        return recon, z, loss
    
    def update_statistics(self, loss: torch.Tensor, ema_alpha: float = 0.99):
        """Update exponential moving averages for online learning."""
        with torch.no_grad():
            self.ema_loss = ema_alpha * self.ema_loss + (1 - ema_alpha) * loss.detach()
            
            # Update variance estimate
            variance = (loss.detach() - self.ema_loss) ** 2
            self.ema_variance = ema_alpha * self.ema_variance + (1 - ema_alpha) * variance
    
    def anomaly_score(self, x: torch.Tensor, threshold_sigma: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly score based on reconstruction error.
        
        Args:
            x: [B, input_dim]
            threshold_sigma: number of standard deviations for anomaly threshold
        Returns:
            anomaly_scores: [B] - reconstruction error
            is_anomaly: [B] - boolean tensor
        """
        recon, _, loss = self.forward(x)
        
        # Per-sample reconstruction error
        error = F.mse_loss(recon, x, reduction='none').mean(dim=1)
        
        # Threshold based on online statistics
        threshold = self.ema_loss + threshold_sigma * torch.sqrt(self.ema_variance)
        is_anomaly = error > threshold
        
        return error, is_anomaly


class OnlineAnomalyDetector(nn.Module):
    """Wrapper for online autoencoder with streaming state management."""
    
    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        self.autoencoder = StreamingAutoencoder(input_dim, latent_dim, hidden_dim)
        self.register_buffer('anomaly_buffer', torch.zeros(100))
        self.buffer_idx = 0
        
    def process_alert(self, alert: torch.Tensor, learn_rate: float = 0.01) -> dict:
        """
        Process single alert with online learning.
        
        Args:
            alert: [input_dim] - single alert features
            learn_rate: learning rate for parameter updates
        Returns:
            {
                'reconstruction': reconstructed alert,
                'latent': latent representation,
                'anomaly_score': reconstruction error,
                'is_anomaly': boolean,
                'loss': reconstruction loss
            }
        """
        if alert.dim() == 1:
            alert = alert.unsqueeze(0)
        
        # Forward pass
        recon, z, loss = self.autoencoder(alert)
        
        # Anomaly detection
        error, is_anomaly = self.autoencoder.anomaly_score(alert)
        
        # Online learning: update statistics
        self.autoencoder.update_statistics(loss)
        
        # Gradient-based parameter update (lightweight)
        loss_val = loss.item()
        if learn_rate > 0 and loss_val > 0:
            loss.backward()
            with torch.no_grad():
                for param in self.autoencoder.parameters():
                    if param.grad is not None:
                        param.data -= learn_rate * param.grad
                        param.grad.zero_()
        
        return {
            'reconstruction': recon.detach(),
            'latent': z.detach(),
            'anomaly_score': error.detach(),
            'is_anomaly': is_anomaly.detach(),
            'loss': loss.detach(),
        }
