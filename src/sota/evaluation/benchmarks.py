import torch
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import precision_recall_curve, auc, f1_score
from typing import Dict, Any, List

from src.sota.models.transformer import AnomalyTransformer
from src.sota.models.tranad import TranAD
from src.sota.models.timesnet import TimesNet
from backend.ml_models import Autoencoder, VariationalAutoencoder, USAD

class AnomalyBenchmarkRunner:
    """
    Scientific Benchmarking Runner for Anomaly Detection.
    Computes professional metrics (AUC-PR, F1-Max) across all versions.
    """
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logging.basicConfig(level=logging.INFO)

    def calculate_metrics(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """
        Calculates academic standard metrics.
        """
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auc_pr = auc(recall, precision)
        
        # F1-Max Calculation
        f1_scores = []
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        
        f1_max = np.max(f1_scores) if len(f1_scores) > 0 else 0
        
        # Recall at top-k (where k is number of anomalies)
        # (This is the 'Discovery Power' metric)
        k = int(np.sum(y_true))
        top_k_indices = np.argsort(y_scores)[-k:]
        discovery_power = np.sum(y_true[top_k_indices]) / k if k > 0 else 0

        return {
            "AUC-PR": float(auc_pr),
            "F1-Max": float(f1_max),
            "Discovery-Power": float(discovery_power)
        }

    def run_benchmark(self, x_test: torch.Tensor, y_test: np.ndarray, model: torch.nn.Module, model_type: str) -> Dict[str, float]:
        """
        Runs a single model through the test set and calculates metrics.
        """
        model.to(self.device).eval()
        x_test = x_test.to(self.device)
        
        with torch.no_grad():
            if model_type in ["autoencoder", "vae"]:
                # Reconstruction error for standard baselines
                if model_type == "autoencoder":
                    out = model(x_test)
                else:
                    out, _, _ = model(x_test)
                scores = torch.mean((out - x_test)**2, dim=1).cpu().numpy()
            
            elif model_type == "transformer":
                out, _, _, _ = model(x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
                
            elif model_type == "tranad":
                _, out = model(x_test, x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
                
            elif model_type == "timesnet":
                out = model(x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
            
            elif model_type == "usad":
                w1, w2 = model(x_test.squeeze(-1))
                # USAD uses alpha=0.5, beta=0.5 for scores usually
                scores = 0.5 * torch.mean((x_test.squeeze(-1) - w1)**2, dim=1) + 0.5 * torch.mean((x_test.squeeze(-1) - w2)**2, dim=1)
                scores = scores.cpu().numpy()
            
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        return self.calculate_metrics(y_test, scores)

    def run_ablation_study(self, x_test: torch.Tensor, y_test: np.ndarray, model_dict: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """
        Systematically tests combinations of experts to prove MoE value.
        """
        from itertools import combinations
        logging.info("Starting Ablation Study...")
        ablation_results = {}
        
        expert_names = list(model_dict.keys())
        
        # Test all combinations from 1 to N
        for r in range(1, len(expert_names) + 1):
            for combo in combinations(expert_names, r):
                combo_name = " + ".join(combo)
                logging.info(f"Evaluating ablation: {combo_name}")
                
                # Compute integrated scores for the combination
                all_combo_scores = []
                for name in combo:
                    model = model_dict[name]
                    model.to(self.device).eval()
                    with torch.no_grad():
                        if name == "transformer":
                            out, _, _, _ = model(x_test.to(self.device))
                            s = torch.mean((out - x_test.to(self.device))**2, dim=(1, 2))
                        elif name == "tranad":
                            _, out = model(x_test.to(self.device), x_test.to(self.device))
                            s = torch.mean((out - x_test.to(self.device))**2, dim=(1, 2))
                        elif name == "timesnet":
                            out = model(x_test.to(self.device))
                            s = torch.mean((out - x_test.to(self.device))**2, dim=(1, 2))
                        all_combo_scores.append(s.cpu().numpy())
                
                integrated_score = np.mean(all_combo_scores, axis=0)
                ablation_results[combo_name] = self.calculate_metrics(y_test, integrated_score)
                
        return ablation_results
    def calculate_bootstrapped_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, n_resamples: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Computes mean and std using bootstrapping (resampling with replacement).
        """
        all_metrics = []
        n = len(y_true)
        
        for _ in range(n_resamples):
            # Resample indices with replacement
            idx = np.random.choice(n, n, replace=True)
            resampled_y_true = y_true[idx]
            resampled_y_scores = y_scores[idx]
            
            # Ensure at least one positive and one negative sample for calculation
            if len(np.unique(resampled_y_true)) < 2:
                continue
                
            metrics = self.calculate_metrics(resampled_y_true, resampled_y_scores)
            all_metrics.append(metrics)
            
        # Convert list of dicts to dict of lists
        cols = {k: [m[k] for m in all_metrics] for k in all_metrics[0].keys()}
        
        # Calculate final stats
        bootstrapped_stats = {}
        for k, v in cols.items():
            bootstrapped_stats[k] = {
                "mean": float(np.mean(v)),
                "std": float(np.std(v))
            }
            
        return bootstrapped_stats

    def run_bootstrapped_benchmark(self, x_test: torch.Tensor, y_test: np.ndarray, model: torch.nn.Module, model_type: str, n_resamples: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Runs a single model and returns bootstrapped confidence intervals.
        """
        model.to(self.device).eval()
        x_test = x_test.to(self.device)
        
        with torch.no_grad():
            if model_type in ["autoencoder", "vae"]:
                # Flatten the window dimension for simple Linear baselines
                x_flat = x_test.squeeze(-1) # (N, L, 1) -> (N, L)
                if model_type == "autoencoder":
                    out = model(x_flat)
                else:
                    out, _, _ = model(x_flat)
                scores = torch.mean((out - x_flat)**2, dim=1).cpu().numpy()
            
            elif model_type == "transformer":
                out, _, _, _ = model(x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
                
            elif model_type == "tranad":
                _, out = model(x_test, x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
                
            elif model_type == "timesnet":
                out = model(x_test)
                scores = torch.mean((out - x_test)**2, dim=(1, 2)).cpu().numpy()
            
            elif model_type == "usad":
                x_flat = x_test.squeeze(-1)
                w1, w2 = model(x_flat)
                scores = 0.5 * torch.mean((x_flat - w1)**2, dim=1) + 0.5 * torch.mean((x_flat - w2)**2, dim=1)
                scores = scores.cpu().numpy()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        return self.calculate_bootstrapped_metrics(y_test, scores, n_resamples)
