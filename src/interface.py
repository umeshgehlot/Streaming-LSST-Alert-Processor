import os
import torch
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from src.sota.models.transformer import AnomalyTransformer
from src.sota.models.tranad import TranAD
from src.sota.models.timesnet import TimesNet
from src.sota.trainer import SotaTrainer
from src.sota.data.datasets import get_dataloader
from src.sota.models.timesnet import TimesNet
from src.sota.trainer import SotaTrainer
from src.sota.data.datasets import get_dataloader
from src.sota.data.streaming import get_streaming_dataloader
from backend.sota_models import StackedEnsembleExpert
from src.sota.evaluation.plotting import plot_anomaly_discovery
from src.sota.agents.reasoning import AstroAgent

class AstroAnomalyEngine:
    """
    The Unified Discovery Engine for Astronomical Anomaly Detection.
    Connects Data Pipelines, SOTA Experts, and AI Analysis.
    """
    def __init__(self, device: str = None, models_dir: str = "models", reports_dir: str = "reports"):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.trainer = SotaTrainer(device=self.device)
        self.agent = AstroAgent() # Initializing Reasoning Agent
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def _is_big_data(self, file_path: str) -> bool:
        """Heuristic to decide between standard and streaming loaders."""
        return os.path.getsize(file_path) > 1024 * 1024 * 500 # 500MB threshold

    def train(self, data_path: str, model_types: List[str] = ["transformer", "tranad", "timesnet"], epochs: int = 5):
        """
        Unified training interface.
        """
        logging.info(f"Initiating training on {data_path}...")
        is_streaming = self._is_big_data(data_path)
        
        # Load data (Lazy mapping)
        if not is_streaming:
            df = pd.read_csv(data_path)
            # Simplistic extraction for demonstration
            data = df['flux'].values.reshape(-1, 1)
            data = (data - data.mean()) / (data.std() + 1e-9)
        
        for m_type in model_types:
            logging.info(f"Training {m_type.upper()} expert...")
            
            # Setup configs
            config = {"epochs": epochs, "batch_size": 256, "lr": 5e-5, "grad_accum_steps": 2}
            if m_type == "tranad": config["win_size"] = 10
            else: config["win_size"] = 32
            
            # Initialize model
            if m_type == "transformer": model = AnomalyTransformer(win_size=32, enc_in=1, c_out=1)
            elif m_type == "tranad": model = TranAD(feats=1, window=10)
            elif m_type == "timesnet": model = TimesNet(enc_in=1, c_out=1, seq_len=32)
            
            # Dataloader selection
            if is_streaming:
                loader = get_streaming_dataloader(data_path, batch_size=config["batch_size"], window_size=config["win_size"])
            else:
                loader = get_dataloader(data, batch_size=config["batch_size"], window_size=config["win_size"])
                
            # Train and Save
            trained_model = self.trainer.train_model(model, loader, config, model_type=m_type)
            self.trainer.save_model(trained_model, f"{m_type}_latest", savedir=self.models_dir)

    def discover(self, data_path: str, weights_suffix: str = "latest") -> Dict[str, Any]:
        """
        Unified discovery interface using the Stacked Ensemble.
        """
        logging.info(f"Running discovery pipeline on {data_path}...")
        
        # Load data for discovery
        df = pd.read_csv(data_path)
        flux_values = df['flux'].values.astype(np.float32)
        
        # Create sliding windows manually for ensemble input
        win_size = 32 # Default for ensemble
        windows = []
        for i in range(len(flux_values) - win_size + 1):
            windows.append(flux_values[i : i + win_size])
        x_tensor = torch.from_numpy(np.array(windows)).float().unsqueeze(-1)
        
        # Initialize ensemble and load weights
        ensemble = StackedEnsembleExpert(input_dim=1, seq_len=win_size, device=self.device)
        
        # Attempt to load trained weights
        for m_name in ["transformer", "tranad", "timesnet"]:
            w_path = os.path.join(self.models_dir, f"{m_name}_{weights_suffix}.pth")
            if os.path.exists(w_path):
                getattr(ensemble, m_name).load_state_dict(torch.load(w_path, map_location=self.device))
                logging.info(f"Loaded weights for {m_name} from {w_path}")

        # Inference
        expert_scores = ensemble(x_tensor)
        unified_score = np.mean(list(expert_scores.values()), axis=0)
        
        # Reasoning on the Top Anomaly
        scientific_interpretation = "No significant anomaly detected for detailed reasoning."
        if results["top_anomalies"]:
            top_idx = results["top_anomalies"][0]
            candidate_info = {
                "id": f"window_{top_idx}",
                "anomaly_score": unified_score[top_idx],
                "features": "Sudden transient detected in temporal sequence" if unified_score[top_idx] > 0.8 else "Subtle periodic variation"
            }
            scientific_interpretation = self.agent.reason_on_discovery(candidate_info)
        
        results["scientific_interpretation"] = scientific_interpretation
        
        return results

    def save_report(self, results: Dict[str, Any], filename: str = "discovery_report.json"):
        path = os.path.join(self.reports_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"Discovery report saved to {path}")
        return path

    def visualize_discovery(self, results: Dict[str, Any], data_path: str, filename: str = "discovery_plot.png"):
        """
        Generates and saves a visual analytics plot for the discovery run.
        """
        logging.info("Generating visual analytics...")
        df = pd.read_csv(data_path)
        flux = df['flux'].values
        
        scores = np.array(results['anomaly_scores'])
        top_anomalies = results['top_anomalies']
        
        out_path = os.path.join(self.reports_dir, filename)
        fig = plot_anomaly_discovery(flux, scores, top_anomalies, output_path=out_path)
        return out_path
