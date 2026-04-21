import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TranAD(nn.Module):
    """
    SOTA TranAD model (VLDB 22) for multi-expert anomaly detection.
    Optimized for adversarial reconstruction.
    """
    def __init__(self, feats, window=10):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.n_feats = feats
        self.n_window = window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        
        # Consistent batch_first=True for all SOTA models
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2 * feats, nhead=min(feats, 8), dim_feedforward=64, dropout=0.1, batch_first=True),
            num_layers=1
        )
        self.transformer_decoder1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=2 * feats, nhead=min(feats, 8), dim_feedforward=64, dropout=0.1, batch_first=True),
            num_layers=1
        )
        self.transformer_decoder2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=2 * feats, nhead=min(feats, 8), dim_feedforward=64, dropout=0.1, batch_first=True),
            num_layers=1
        )
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src_c = torch.cat((src, c), dim=2)
        src_c = src_c * math.sqrt(self.n_feats)
        src_c = self.pos_encoder(src_c)
        memory = self.transformer_encoder(src_c)
        tgt_r = tgt.repeat(1, 1, 2)
        return tgt_r, memory

    def forward(self, src, tgt):
        # Phase 1 - Basic Reconstruction
        c = torch.zeros_like(src)
        tgt_r, memory = self.encode(src, c, tgt)
        x1 = self.fcn(self.transformer_decoder1(tgt_r, memory))
        # Phase 2 - Adversarial Focus
        c = (x1 - src) ** 2
        tgt_r, memory = self.encode(src, c, tgt)
        x2 = self.fcn(self.transformer_decoder2(tgt_r, memory))
        return x1, x2
