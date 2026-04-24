"""
Configuration file for Streaming LSST Alert Processor.
Customize this to adjust model parameters and processing settings.
"""

import torch

# ============================================================================
# Device Configuration
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_HALF_PRECISION = False  # Use float16 for faster inference on GPU

# ============================================================================
# Data Pipeline Configuration
# ============================================================================

FEATURE_EXTRACTION = {
    'feature_dim': 16,              # Number of features extracted per alert
    'normalize': True,              # Online normalization
    'ema_alpha': 0.99,              # Exponential moving average factor
    'warmup_samples': 1000,         # Samples before stats stabilize
}

PIPELINE_BUFFER = {
    'buffer_size': 100,             # Max alerts in buffer
    'batch_size': 32,               # Batch size for inference
}

# ============================================================================
# Streaming Transformer Configuration
# ============================================================================

STREAMING_TRANSFORMER = {
    'enabled': True,
    'input_dim': 16,
    'd_model': 64,                  # Hidden dimension
    'n_layers': 2,                  # Number of attention layers
    'n_heads': 4,                   # Number of attention heads
    'd_ff': 256,                    # Feed-forward hidden dimension
    'window_size': 32,              # Attention context window
    'output_dim': 32,               # Output embedding dimension
    'dropout': 0.1,
}

# ============================================================================
# Online Autoencoder Configuration
# ============================================================================

ONLINE_AUTOENCODER = {
    'enabled': True,
    'input_dim': 16,
    'latent_dim': 8,                # Latent representation size
    'hidden_dim': 32,               # Hidden layer dimension
    'learn_rate': 0.01,             # Online learning rate
    'threshold_sigma': 4.0,         # Increased from 3.0 to improve precision
}

# ============================================================================
# Streaming GNN Configuration
# ============================================================================

STREAMING_GNN = {
    'enabled': True,
    'in_dim': 16,
    'hidden_dim': 32,
    'out_dim': 32,
    'n_layers': 2,
    'dropout': 0.1,
    'max_nodes': 1000,              # Increased for better spatial context
    'spatial_threshold': 1.0,       # Increased to 1.0 degree to capture cluster context
    'edge_weight': 0.3,             # Reduced weight for each neighbor to avoid noise
}

# ============================================================================
# Alert Simulator Configuration
# ============================================================================

ALERT_SIMULATOR = {
    'simulator_type': 'standard',   # 'standard', 'high_variability', or 'burst'
    'seed': 42,
    'anomaly_rate': 0.1,            # Fraction of anomalous alerts
    'base_rate_hz': 100.0,          # Base alert rate
}

# ============================================================================
# Advanced Configuration
# ============================================================================

ADVANCED = {
    # Anomaly detection
    'anomaly': {
        'ema_alpha': 0.99,          # EMA factor for statistics
        'min_samples': 100,         # Minimum samples before anomaly detection
    },
    
    # GNN processing
    'gnn': {
        'edge_timeout': 3600,       # Remove edges older than this (seconds)
        'node_timeout': 3600,       # Remove nodes older than this
        'max_neighbors': 10,        # Max edges per new node
    },
}

# ============================================================================
# Presets and Getters
# ============================================================================

def get_config(preset: str = 'balanced') -> dict:
    config = {
        'device': DEVICE,
        'feature_extraction': FEATURE_EXTRACTION,
        'pipeline_buffer': PIPELINE_BUFFER,
        'streaming_transformer': STREAMING_TRANSFORMER.copy(),
        'online_autoencoder': ONLINE_AUTOENCODER.copy(),
        'streaming_gnn': STREAMING_GNN.copy(),
        'advanced': ADVANCED,
    }
    return config
