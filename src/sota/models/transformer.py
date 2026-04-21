import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt

# --- EMBEDDING ---
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))

# --- ATTENTION ---
class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.distances = torch.zeros((win_size, win_size))
        for i in range(win_size):
            for j in range(win_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn = scale * scores
        sigma = sigma.transpose(1, 2)
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).to(queries.device)
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))
        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)
        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        H = self.n_heads
        x = queries
        qs = self.query_projection(queries).view(B, L, H, -1)
        ks = self.key_projection(keys).view(B, L, H, -1)
        vs = self.value_projection(values).view(B, L, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)
        out, series, prior, sigma = self.inner_attention(qs, ks, vs, sigma, attn_mask)
        return self.out_projection(out.view(B, L, -1)), series, prior, sigma

# --- CORE MODEL ---
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, attn_mask=None):
        nx, attn, mask, sigma = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(nx)
        y = x = self.norm1(x)
        y = self.dropout(F.relu(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn, mask, sigma

class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512, dropout=0.0):
        super(AnomalyTransformer, self).__init__()
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.encoder = nn.ModuleList([
            EncoderLayer(
                AttentionLayer(AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=True), d_model, n_heads),
                d_model, d_ff, dropout=dropout
            ) for _ in range(e_layers)
        ])
        self.projection = nn.Linear(d_model, c_out, bias=True)
    def forward(self, x):
        enc_out = self.embedding(x)
        series_list, prior_list, sigma_list = [], [], []
        for layer in self.encoder:
            enc_out, series, prior, sigma = layer(enc_out)
            series_list.append(series); prior_list.append(prior); sigma_list.append(sigma)
        return self.projection(enc_out), series_list, prior_list, sigma_list
