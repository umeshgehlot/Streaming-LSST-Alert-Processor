"""
System orchestrator that integrates all streaming components.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from .models.streaming_transformer import StreamingTransformer
from .models.online_autoencoder import OnlineAnomalyDetector
from .models.streaming_gnn import StreamingGNN, StreamingAlertGraph
from .pipeline.data_pipeline import StreamingPipeline
from .config import get_config
import numpy as np
import os


class StreamingLSSTProcessor:
    """Main orchestrator for streaming LSST alert processing."""
    
    def __init__(self, device: str = 'cpu', enable_transformer: bool = True, 
                 enable_autoencoder: bool = True, enable_gnn: bool = True):
        self.device = torch.device(device)
        self.config = get_config()
        self.feature_dim = self.config['feature_extraction']['feature_dim']
        
        self.pipeline = StreamingPipeline(
            feature_dim=self.feature_dim, 
            buffer_size=self.config['pipeline_buffer']['buffer_size']
        )
        
        self.transformer = None
        self.anomaly_detector = None
        self.graph_processor = None
        
        if enable_transformer:
            trans_cfg = self.config['streaming_transformer']
            self.transformer = StreamingTransformer(
                input_dim=self.feature_dim,
                d_model=trans_cfg['d_model'],
                n_layers=trans_cfg['n_layers'],
                n_heads=trans_cfg['n_heads'],
                d_ff=trans_cfg['d_ff'],
                window_size=trans_cfg['window_size'],
                output_dim=trans_cfg['output_dim'],
                dropout=trans_cfg['dropout']
            ).to(self.device)
        
        if enable_autoencoder:
            ae_cfg = self.config['online_autoencoder']
            self.anomaly_detector = OnlineAnomalyDetector(
                input_dim=self.feature_dim,
                latent_dim=ae_cfg['latent_dim'],
                hidden_dim=ae_cfg['hidden_dim']
            ).to(self.device)
            self.ae_threshold_sigma = ae_cfg.get('threshold_sigma', 3.0)
            self.ae_learn_rate = ae_cfg.get('learn_rate', 0.01)
        
        if enable_gnn:
            gnn_cfg = self.config['streaming_gnn']
            gnn = StreamingGNN(in_dim=self.feature_dim, hidden_dim=gnn_cfg['hidden_dim'], out_dim=gnn_cfg['out_dim'])
            self.graph_processor = StreamingAlertGraph(gnn, self.feature_dim, max_nodes=gnn_cfg['max_nodes'])
            self.spatial_threshold = gnn_cfg.get('spatial_threshold', 1.0)
            self.edge_weight = gnn_cfg.get('edge_weight', 0.5)
            
        self._load_pretrained_models()

    def _load_pretrained_models(self):
        """Load pretrained models from disk if they exist."""
        import pathlib
        project_root = pathlib.Path(__file__).parent.parent
        trained_dir = project_root / "streaming_lsst" / "trained_models"
        
        ae_path = trained_dir / "autoencoder_trained.pt"
        if self.anomaly_detector and ae_path.exists():
            try:
                self.anomaly_detector.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
                print(f"Loaded pretrained Autoencoder from {ae_path}")
            except Exception as e:
                print(f"Failed to load pretrained Autoencoder: {e}")
                
        trans_path = trained_dir / "transformer_trained.pt"
        if self.transformer and trans_path.exists():
            try:
                # The saved transformer is a ModuleDict with batch-mode components. 
                # We can map them back to the streaming model's components:
                state_dict = torch.load(trans_path, map_location=self.device)
                
                # We need to map state dict keys from the batch model to the streaming model
                new_state_dict = {}
                # This depends on exact module names. For now we skip auto-loading transformer 
                # because the architectures (streaming vs batch) differ slightly in structure
                # but we've successfully loaded the autoencoder which is identical.
            except Exception as e:
                pass
    
    def process_alert(self, alert: Dict) -> Dict:
        import time
        start_time = time.perf_counter()
        
        # 1. Feature extraction
        features, alert_id, extract_time = self.pipeline.process_alert(alert)
        features = features.to(self.device)
        
        result = {
            'alert_id': alert_id,
            'features': features.cpu().detach(),
            'extract_time_ms': extract_time,
        }
        
        # 2. Autoencoder
        if self.anomaly_detector:
            with torch.no_grad():
                recon, _, _ = self.anomaly_detector.autoencoder(features.unsqueeze(0))
                error = torch.mean((recon - features.unsqueeze(0))**2).item()
                threshold = self.anomaly_detector.autoencoder.ema_loss + \
                            self.ae_threshold_sigma * torch.sqrt(self.anomaly_detector.autoencoder.ema_variance)
            
            update_lr = self.ae_learn_rate if error < threshold * 3.0 else 0.0
            anomaly_result = self.anomaly_detector.process_alert(features, learn_rate=update_lr)
            
            result['anomaly_score'] = anomaly_result['anomaly_score'].item()
            result['is_anomaly'] = (anomaly_result['anomaly_score'] > threshold).item()
            result['anomaly_details'] = {'reconstruction_error': error, 'threshold': threshold.item()}
        
        # 3. GNN
        if self.graph_processor:
            if 'alert' in alert:
                ra, dec = alert['alert'].get('ra', 0.0), alert['alert'].get('dec', 0.0)
            else:
                cand = alert.get('candidate', {})
                ra, dec = cand.get('ra', 0.0), cand.get('dec', 0.0)
            
            self.graph_processor.coords[alert_id] = (ra, dec)
            self.graph_processor.update_node(alert_id, features.cpu())
            
            recent_ids = list(self.graph_processor.coords.keys())[-300:]
            for other_id in recent_ids:
                if other_id == alert_id: continue
                ora, odec = self.graph_processor.coords[other_id]
                dist = np.sqrt((ra-ora)**2 + (dec-odec)**2)
                if dist < self.spatial_threshold:
                    self.graph_processor.add_relation(alert_id, other_id, weight=self.edge_weight)
                    self.graph_processor.add_relation(other_id, alert_id, weight=self.edge_weight)

            _, gnn_context = self.graph_processor.compute_embeddings(alert_id)
            
            if torch.norm(gnn_context) > 0:
                # Slice gnn_context to match features dimension
                context_diff = torch.norm(gnn_context[:self.feature_dim] - features.cpu()).item()
                result['context_anomaly_score'] = context_diff
                result['anomaly_score'] = result['anomaly_score'] * 20.0 + context_diff * 5.0
                if context_diff > 4.0 or result['anomaly_score'] > 15.0:
                    result['is_anomaly'] = True
        
        result['latency_ms'] = (time.perf_counter() - start_time) * 1000
        return result

    def get_pipeline_stats(self) -> Dict:
        return self.pipeline.get_metrics()
    
    def reset_models(self):
        if self.graph_processor:
            self.graph_processor = StreamingAlertGraph(self.graph_processor.gnn, self.feature_dim)
