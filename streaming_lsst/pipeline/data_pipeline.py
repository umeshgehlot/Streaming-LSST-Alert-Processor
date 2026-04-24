"""
Streaming data pipeline for LSST alert processing.
Handles real-time feature extraction, normalization, and buffering.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import time


class StreamingAlertFeatureExtractor:
    """Extract and normalize features from LSST alerts in real-time."""
    
    # LSST alert feature names (subset of key features)
    FEATURE_NAMES = [
        'ra', 'dec',  # Position
        'mag', 'magErr',  # Magnitude and error
        'flux', 'fluxErr',  # Flux and error
        'objectId',  # Object identifier
        'candidate_jd',  # Julian date
        'candidate_ndethist',  # Number of detections
        'candidate_ncandgn',  # Number of candidates
        'candidate_nnegn',  # Number of negatives
        'candidate_diffmaglim',  # Difference magnitude limit
        'candidate_ranr',  # RA of nearest reference
        'candidate_decnr',  # Dec of nearest reference
    ]
    
    def __init__(self, feature_dim: int = 16, normalize: bool = True):
        self.feature_dim = feature_dim
        self.normalize = normalize
        
        # Running statistics for normalization
        self.mean = torch.zeros(feature_dim)
        self.std = torch.ones(feature_dim)
        self.update_count = 0
        self.ema_alpha = 0.99
    
    def extract_features(self, alert: Dict) -> np.ndarray:
        """
        Extract key features from alert dictionary.
        Supports both internal LSST simulator and ZTF-like schemas.
        """
        features = []
        
        # Determine schema
        if 'alert' in alert:
            # Internal/LSST Simulator Schema
            alert_data = alert.get('alert', {})
            candidate = alert_data.get('candidate', {})
            ra = alert_data.get('ra', 0.0)
            dec = alert_data.get('dec', 0.0)
        else:
            # ZTF-like Schema (candidate at top level)
            candidate = alert.get('candidate', {})
            ra = candidate.get('ra', 0.0)
            dec = candidate.get('dec', 0.0)
            
        # Unified feature extraction
        features.append(float(ra))
        features.append(float(dec))
        
        # Magnitude (magpsf in ZTF, mag in LSST)
        features.append(float(candidate.get('magpsf', candidate.get('mag', 0.0))))
        features.append(float(candidate.get('sigmapsf', candidate.get('magerr', 0.0))))
        
        # Flux (flux in LSST, fallback to 0 for ZTF if not present)
        features.append(float(candidate.get('flux', 0.0)))
        features.append(float(candidate.get('fluxerr', 0.0)))
        
        # Historical
        features.append(float(candidate.get('ndethist', 0.0)))
        features.append(float(candidate.get('ncandgn', candidate.get('ncovhist', 0.0))))
        features.append(float(candidate.get('nnegn', 0.0)))
        
        # Others
        features.append(float(candidate.get('ranr', 0.0)))
        features.append(float(candidate.get('decnr', 0.0)))
        features.append(float(candidate.get('distpsnr1', 0.0)))
        features.append(float(candidate.get('rmag', 0.0)))
        features.append(float(candidate.get('imag', 0.0)))
        features.append(float(candidate.get('zmag', 0.0)))
        features.append(float(candidate.get('sgscore1', candidate.get('sgscore', 0.0))))
        
        features = np.array(features[:self.feature_dim], dtype=np.float32)
        
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
            
        return features
    
    def normalize_features(self, features: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize features using running statistics."""
        
        features_tensor = torch.from_numpy(features).float()
        
        if update_stats and self.update_count < 1000:  # Only update for first 1000 samples
            # Online mean/std estimation
            with torch.no_grad():
                self.mean = self.ema_alpha * self.mean + (1 - self.ema_alpha) * features_tensor
                delta = features_tensor - self.mean
                self.std = torch.sqrt(
                    self.ema_alpha * self.std**2 + (1 - self.ema_alpha) * delta**2
                )
                self.update_count += 1
        
        # Normalize
        normalized = (features_tensor - self.mean) / (self.std + 1e-8)
        return normalized.numpy()


class StreamingAlertBuffer:
    """Efficient buffer for streaming alerts with windowing."""
    
    def __init__(self, window_size: int = 100, feature_dim: int = 16):
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        self.buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.alert_ids = deque(maxlen=window_size)
    
    def add_alert(self, alert_id: str, features: np.ndarray, timestamp: float):
        """Add alert to buffer."""
        self.buffer.append(features.copy())
        self.timestamps.append(timestamp)
        self.alert_ids.append(alert_id)
    
    def get_batch(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, List[str]]:
        """Get current buffer as batch tensor."""
        if not self.buffer:
            return torch.zeros(0, self.feature_dim), []
        
        size = batch_size or len(self.buffer)
        if size > len(self.buffer):
            size = len(self.buffer)
        
        features = torch.from_numpy(np.array(list(self.buffer)[-size:])).float()
        ids = list(self.alert_ids)[-size:]
        
        return features, ids
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()
        self.timestamps.clear()
        self.alert_ids.clear()


class StreamingPipeline:
    """Main streaming pipeline combining extraction, buffering, and preprocessing."""
    
    def __init__(self, feature_dim: int = 16, buffer_size: int = 100, 
                 batch_size: int = 32, normalize: bool = True):
        self.feature_dim = feature_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        self.extractor = StreamingAlertFeatureExtractor(feature_dim, normalize)
        self.buffer = StreamingAlertBuffer(buffer_size, feature_dim)
        
        # Timing statistics
        self.timings = deque(maxlen=1000)
        self.throughput_timestamps = deque(maxlen=100)
    
    def process_alert(self, alert: Dict) -> Tuple[torch.Tensor, str, float]:
        """
        Process single alert end-to-end.
        
        Args:
            alert: Alert dictionary
        Returns:
            (features, alert_id, processing_time_ms)
        """
        start_time = time.perf_counter()
        
        # Extract features
        features = self.extractor.extract_features(alert)
        
        # Normalize
        features = self.extractor.normalize_features(features, update_stats=True)
        
        # Add to buffer
        alert_id = str(alert.get('alert', {}).get('objectId', 
                      alert.get('candidate', {}).get('objectid', 
                      alert.get('alertId', 'unknown'))))
        self.buffer.add_alert(alert_id, features, start_time)
        
        # Record timing
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.timings.append(elapsed_ms)
        self.throughput_timestamps.append(start_time)
        
        return torch.from_numpy(features).float(), alert_id, elapsed_ms
    
    def get_batch(self) -> Tuple[torch.Tensor, List[str]]:
        """Get current buffer batch."""
        return self.buffer.get_batch(self.batch_size)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get pipeline performance metrics."""
        if not self.timings:
            return {'avg_latency_ms': 0.0, 'throughput_alerts_per_sec': 0.0}
        
        timings = list(self.timings)
        
        # Compute throughput (alerts in last second)
        now = time.perf_counter()
        recent_alerts = sum(1 for t in self.throughput_timestamps if now - t < 1.0)
        
        return {
            'avg_latency_ms': np.mean(timings),
            'p95_latency_ms': np.percentile(timings, 95),
            'p99_latency_ms': np.percentile(timings, 99),
            'max_latency_ms': np.max(timings),
            'throughput_alerts_per_sec': recent_alerts,
            'buffer_occupancy': len(self.buffer.buffer) / self.buffer_size,
        }
