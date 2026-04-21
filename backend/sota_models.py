import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.sota.models.transformer import AnomalyTransformer
from src.sota.models.tranad import TranAD
from src.sota.models.timesnet import TimesNet

class GatingNetwork(nn.Module):
    """
    Symmetry-Aware Gating Network (SAG-Net).
    Computes input-dependent weights for expert routing.
    """
    def __init__(self, input_dim: int, window_size: int, num_experts: int = 3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim * window_size, 64)
        self.fc2 = nn.Linear(64, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: (Batch, Window, Dim)
        x_flat = self.flatten(x)
        h = F.relu(self.fc1(x_flat))
        logits = self.fc2(h)
        weights = self.softmax(logits)
        return weights

class SotaExpertFactory:
    """
    Factory to create and load weights for SOTA experts.
    """
    @staticmethod
    def load_expert(model_type: str, input_dim: int, window_size: int, weights_path: str = None, device: str = 'cpu'):
        if model_type == 'transformer':
            model = AnomalyTransformer(win_size=window_size, enc_in=input_dim, c_out=input_dim)
        elif model_type == 'tranad':
            model = TranAD(feats=input_dim, window=window_size)
        elif model_type == 'timesnet':
            model = TimesNet(enc_in=input_dim, c_out=input_dim, seq_len=window_size)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        if weights_path:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        
        return model.to(device).eval()

class StackedEnsembleExpert(nn.Module):
    """
    Orthogonal Ensemble integrating Anomaly Transformer, TranAD, and TimesNet
    with a Dynamic Gating Mechanism (SAG-Net).
    """
    def __init__(self, input_dim: int, seq_len: int = 32, device: str = None):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.seq_len = seq_len
        
        # 1. Experts
        self.transformer = AnomalyTransformer(win_size=seq_len, enc_in=input_dim, c_out=input_dim).to(self.device)
        self.tranad = TranAD(feats=input_dim, window=seq_len).to(self.device)
        self.timesnet = TimesNet(enc_in=input_dim, c_out=input_dim, seq_len=seq_len).to(self.device)
        
        # 2. Dynamic Gating Mechanism (SAG-Net)
        self.gate = GatingNetwork(input_dim=input_dim, window_size=seq_len, num_experts=3).to(self.device)

    def forward(self, x):
        """
        Computes weighted reconstruction errors using dynamic gating.
        """
        x = x.to(self.device)
        batch_size = x.shape[0]
        
        # A. Compute individual expert errors
        with torch.no_grad():
            out_t, _, _, _ = self.transformer(x)
            err_t = torch.mean((out_t - x) ** 2, dim=(1, 2))
            
            _, out_tr = self.tranad(x, x)
            err_tr = torch.mean((out_tr - x) ** 2, dim=(1, 2))
            
            out_ti = self.timesnet(x)
            err_ti = torch.mean((out_ti - x) ** 2, dim=(1, 2))
        
        # B. Compute Dynamic Weights via Gating Network
        weights = self.gate(x) # (Batch, 3)
        
        # C. Weighted Fusion
        expert_errors = torch.stack([err_t, err_tr, err_ti], dim=1) # (Batch, 3)
        weighted_score = torch.sum(expert_errors * weights, dim=1) # (Batch,)
        
        return {
            "expert_errors": {
                "transformer": err_t.cpu().numpy(),
                "tranad": err_tr.cpu().numpy(),
                "timesnet": err_ti.cpu().numpy()
            },
            "weights": weights.cpu().detach().numpy(),
            "unified_score": weighted_score.cpu().detach().numpy()
        }

    def compute_weighted_anomaly_score(self, forward_output: dict) -> float:
        """
        Extracts the unified score from the forward loop.
        """
        return float(np.mean(forward_output["unified_score"]))
