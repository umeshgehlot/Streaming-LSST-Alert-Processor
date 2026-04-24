"""
Baseline models for anomaly detection comparison.
Includes Isolation Forest and simple non-streaming RNN.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Optional

class IsolationForestBaseline:
    """Random Forest baseline using Isolation Forest for anomaly detection."""
    
    def __init__(self, n_estimators=100, contamination=0.1):
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination,
            random_state=42
        )
        self.is_fitted = False
        self.buffer = []
        self.max_buffer_size = 500 # Size for initial fitting
        
    def process_alert(self, features: torch.Tensor) -> Dict:
        """Process alert using Isolation Forest."""
        import time
        start_time = time.perf_counter()
        
        feat_np = features.detach().cpu().numpy().reshape(1, -1)
        
        # Isolation Forest is typically not "online" in a streaming sense,
        # so we fit on a rolling buffer or initial window.
        if not self.is_fitted:
            self.buffer.append(feat_np[0])
            if len(self.buffer) >= self.max_buffer_size:
                self.model.fit(np.array(self.buffer))
                self.is_fitted = True
            
            # Default to not-anomaly during fitting phase
            latency = (time.perf_counter() - start_time) * 1000
            return {
                'anomaly_score': 0.5,
                'is_anomaly': False,
                'latency_ms': latency
            }
        
        # Predict: 1 for inlier, -1 for outlier
        pred = self.model.predict(feat_np)
        score = -self.model.decision_function(feat_np)[0] # Raw anomaly score
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            'anomaly_score': float(score),
            'is_anomaly': bool(pred[0] == -1),
            'latency_ms': latency
        }

class RNNBaseline(nn.Module):
    """Simple non-streaming GRU-based autoencoder baseline."""
    
    def __init__(self, input_dim=16, hidden_dim=64, n_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.encoder = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Simple online statistics for anomaly thresholding
        self.ema_loss = 0.0
        self.ema_variance = 1.0
        self.alpha = 0.05
        
    def forward(self, x):
        # Expecting x as (batch, seq_len, input_dim)
        _, h = self.encoder(x)
        # Use last hidden state to reconstruct the current frame
        out = self.decoder(h[-1])
        return out
        
    def process_alert(self, features: torch.Tensor) -> Dict:
        """Process alert using standard RNN reconstruction error."""
        import time
        start_time = time.perf_counter()
        
        # Non-streaming RNN needs a sequence, but here we provide it with a single frame
        # to compare direct latency/accuracy in a "stateless" or "simple history" way.
        x = features.unsqueeze(0).unsqueeze(0) # (1, 1, dim)
        
        self.eval()
        with torch.no_grad():
            reconstruction = self.forward(x)
            loss = torch.mean((reconstruction - features)**2).item()
            
        # Update online statistics
        if self.ema_loss == 0:
            self.ema_loss = loss
        else:
            self.ema_loss = (1 - self.alpha) * self.ema_loss + self.alpha * loss
            diff = (loss - self.ema_loss)**2
            self.ema_variance = (1 - self.alpha) * self.ema_variance + self.alpha * diff
            
        # Anomaly detection based on Z-score
        std = np.sqrt(self.ema_variance) + 1e-6
        z_score = (loss - self.ema_loss) / std
        is_anomaly = z_score > 3.0 # 3-sigma rule
        
        latency = (time.perf_counter() - start_time) * 1000
        
        return {
            'anomaly_score': float(loss),
            'is_anomaly': bool(is_anomaly),
            'latency_ms': latency
        }
