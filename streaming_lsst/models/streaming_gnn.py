"""
Lightweight Graph Neural Network for processing alert relationships in streams.
Supports sparse graph updates and efficient message passing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class StreamingGraphLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_mlp = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(), nn.Dropout(dropout))
        self.message_mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        with torch.no_grad():
            if in_dim == out_dim:
                nn.init.eye_(self.node_mlp[0].weight)
                nn.init.eye_(self.message_mlp.weight)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_features.size(0)
        x = self.node_mlp(node_features)
        if edge_index.size(1) == 0: return self.norm(x), torch.zeros_like(x)
        
        src, dst = edge_index[0], edge_index[1]
        messages = self.message_mlp(node_features[src])
        if edge_weights is not None: messages = messages * edge_weights.unsqueeze(1)
            
        aggregated = torch.zeros(num_nodes, self.out_dim, device=node_features.device)
        aggregated.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.out_dim), messages)
        
        counts = torch.zeros(num_nodes, 1, device=node_features.device)
        counts.scatter_add_(0, dst.unsqueeze(1), torch.ones_like(dst.unsqueeze(1).float()))
        context = aggregated / (counts + 1e-8)
        
        return self.norm(x + context), context


class StreamingGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([StreamingGraphLayer(in_dim, out_dim)])
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor, edge_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        x, context = self.layers[0](node_features, edge_index, edge_weights)
        return x, context


class StreamingAlertGraph:
    def __init__(self, gnn_model: StreamingGNN, node_dim: int, max_nodes: int = 1000):
        self.gnn = gnn_model
        self.node_dim = node_dim
        self.max_nodes = max_nodes
        self.node_features = torch.zeros(0, node_dim)
        self.edges_dict = {}
        self.node_ids = []
        self.id_to_idx = {}
        self.coords = {}
    
    def update_node(self, alert_id: str, features: torch.Tensor):
        """Add or update node without computing embeddings."""
        if alert_id not in self.id_to_idx:
            if len(self.node_features) >= self.max_nodes: self._remove_node(0)
            idx = len(self.node_features)
            self.node_ids.append(alert_id)
            self.id_to_idx[alert_id] = idx
            self.node_features = torch.cat([self.node_features, features.unsqueeze(0)], dim=0)
        else:
            self.node_features[self.id_to_idx[alert_id]] = features

    def compute_embeddings(self, alert_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute embeddings for all nodes and return for specific alert."""
        if alert_id not in self.id_to_idx: return torch.zeros(self.gnn.layers[0].out_dim), torch.zeros(self.gnn.layers[0].out_dim)
        edge_index = self._build_edge_index()
        embeddings, context = self.gnn(self.node_features, edge_index)
        idx = self.id_to_idx[alert_id]
        return embeddings[idx], context[idx]
    
    def add_relation(self, alert_id1: str, alert_id2: str, weight: float = 1.0):
        if alert_id1 in self.id_to_idx and alert_id2 in self.id_to_idx:
            self.edges_dict[(self.id_to_idx[alert_id1], self.id_to_idx[alert_id2])] = weight
    
    def _remove_node(self, idx: int):
        aid = self.node_ids.pop(idx)
        del self.id_to_idx[aid]
        if aid in self.coords: del self.coords[aid]
        self.node_features = torch.cat([self.node_features[:idx], self.node_features[idx+1:]], dim=0)
        self.id_to_idx = {aid: i for i, aid in enumerate(self.node_ids)}
        new_edges = {}
        for (s, d), w in self.edges_dict.items():
            if s == idx or d == idx: continue
            new_edges[(s if s < idx else s - 1, d if d < idx else d - 1)] = w
        self.edges_dict = new_edges

    def _build_edge_index(self) -> torch.Tensor:
        if not self.edges_dict: return torch.zeros(2, 0, dtype=torch.long)
        return torch.tensor(list(self.edges_dict.keys()), dtype=torch.long).T
