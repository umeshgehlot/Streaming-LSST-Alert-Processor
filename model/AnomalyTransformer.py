import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

# ------------------------------------------------------------------
#  RoPE utilities
# ------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).type_as(inv_freq)
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('cos', freqs.cos(), persistent=False)
        self.register_buffer('sin', freqs.sin(), persistent=False)

    def forward(self, seq_len):
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(2)  # [1, L, 1, D//2]
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(2)
        return cos, sin

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos[..., :q.shape[-1]]
    sin = sin[..., :q.shape[-1]]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms).type_as(x) * self.weight

# ------------------------------------------------------------------
#  MULTI-SCALE PATCH EMBEDDING (Crucial for Benchmark Accuracy)
# ------------------------------------------------------------------
class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, enc_in, d_model, patch_sizes=(1, 3, 5), dropout=0.1):
        super().__init__()
        assert d_model % len(patch_sizes) == 0
        sub_dim = d_model // len(patch_sizes)
        self.patch_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.ConstantPad1d((ps // 2, ps // 2), 0.0),
                nn.Conv1d(enc_in, sub_dim, kernel_size=ps, stride=1, bias=False),
            ) for ps in patch_sizes
        ])
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        parts = [emb(x_t).transpose(1, 2) for emb in self.patch_embeddings]
        x = torch.cat(parts, dim=-1)
        return self.dropout(self.norm(self.merge(x)))

# ------------------------------------------------------------------
#  MULTI-SCALE ANOMALY ATTENTION (Upgraded from Single-Scale)
# ------------------------------------------------------------------
class MultiScaleAnomalyAttention(nn.Module):
    def __init__(self, win_size, n_heads, d_model, n_scale=3, 
                 attention_dropout=0.1, output_attention=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_scale = n_scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.rope = RotaryEmbedding(self.d_head)

        # Multi-scale learnable parameters (Crucial for beating baseline)
        self.sigma_base = nn.Parameter(torch.randn(n_heads, n_scale) * 0.5 + 1.0)
        self.sigma_weight = nn.Parameter(torch.zeros(n_heads, n_scale))
        self.log_temperature = nn.Parameter(torch.zeros(n_heads, 1, 1))

        # Cache distance matrix to save massive compute
        dists = torch.arange(win_size).unsqueeze(0) - torch.arange(win_size).unsqueeze(1)
        self.register_buffer("distances", dists.float(), persistent=False)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _compute_multi_scale_prior(self, B, L):
        dists = self.distances[:L, :L]
        sigma = F.softplus(self.sigma_base)
        weights = F.softmax(self.sigma_weight, dim=-1)
        
        dists_ = dists.unsqueeze(0).unsqueeze(0)
        sigma_ = sigma.unsqueeze(-1).unsqueeze(-1)
        
        gauss = torch.exp(-0.5 * (dists_ / sigma_) ** 2)
        gauss = gauss / (gauss.sum(dim=-1, keepdim=True) + 1e-8)
        
        prior = (gauss * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        return prior.unsqueeze(0).expand(B, -1, -1, -1)

    def forward(self, x, attn_mask=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        Q = self.W_Q(x).view(B, L, H, D).transpose(1, 2)
        K = self.W_K(x).view(B, L, H, D).transpose(1, 2)
        V = self.W_V(x).view(B, L, H, D).transpose(1, 2)

        cos, sin = self.rope(L)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        temperature = self.log_temperature.exp()
        series = torch.matmul(Q, K.transpose(-2, -1)) / temperature

        if attn_mask is not None:
            series = series.masked_fill(attn_mask == 0, -1e9)

        series = F.softmax(series, dim=-1)
        series = self.dropout(series)

        prior = self._compute_multi_scale_prior(B, L)
        sigma_out = F.softplus(self.sigma_base).detach()

        out = torch.matmul(series, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.out_proj(out)

        if self.output_attention:
            return out, series, prior, sigma_out
        return out,

# ------------------------------------------------------------------
#  ENCODER LAYER (SwiGLU with DWConv for local context)
# ------------------------------------------------------------------
class NextGenEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, win_size, n_scale=3, d_ff=None, dropout=0.1, attention_dropout=0.0):
        super().__init__()
        d_ff = d_ff or int(8 / 3 * d_model) # Standard SwiGLU ratio
        
        self.attention = MultiScaleAnomalyAttention(
            win_size, n_heads, d_model, n_scale, attention_dropout, output_attention=True
        )
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # SwiGLU FFN with Depthwise Conv (Injects local context since no local stream)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dwconv = nn.Conv1d(d_ff, d_ff, 3, padding=1, groups=d_ff)

    def forward(self, x, attn_mask=None):
        residual = x
        x_norm = self.norm1(x)
        x_attn, series, prior, sigma = self.attention(x_norm, attn_mask=attn_mask)
        x = residual + self.dropout(x_attn)

        residual = x
        x_norm = self.norm2(x)
        gate = F.silu(self.w1(x_norm))
        gate = self.dwconv(gate.transpose(1, 2)).transpose(1, 2) # Local context injection
        x = residual + self.dropout(self.w2(gate * self.w3(x_norm)))

        return x, series, prior, sigma

# ------------------------------------------------------------------
#  Anomaly Transformer V2.0 Single-Stream MAX
# ------------------------------------------------------------------
class AnomalyTransformerV2(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, 
                 e_layers=3, d_ff=None, dropout=0.1, n_scale=3, attention_dropout=0.0):
        super().__init__()
        self.n_scale = n_scale
        
        # Upgraded Embedding
        self.embedding = MultiScalePatchEmbedding(enc_in, d_model, (1, 3, 5), dropout)

        self.encoder = nn.ModuleList([
            NextGenEncoderLayer(d_model, n_heads, win_size, n_scale,
                                d_ff=d_ff, dropout=dropout,
                                attention_dropout=attention_dropout)
            for _ in range(e_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def compute_association_discrepancy(self, series_list, prior_list):
        """Mandatory for benchmark F1 scores. Uses Symmetrized KL-Divergence."""
        layer_discs = []
        for series, prior in zip(series_list, prior_list):
            s = series + 1e-8
            p = prior + 1e-8
            
            kl_sp = F.kl_div(s.log(), p, reduction='none').sum(dim=-1).mean(dim=1) # [B, L]
            kl_ps = F.kl_div(p.log(), s, reduction='none').sum(dim=-1).mean(dim=1) # [B, L]
            
            disc = (kl_sp + kl_ps) / 2.0 
            layer_discs.append(disc)
            
        return torch.stack(layer_discs, dim=0).sum(dim=0) # [B, L]

    def forward(self, x, attn_mask=None):
        enc_out = self.embedding(x)
        series_list, prior_list, sigma_list = [], [], []

        for layer in self.encoder:
            enc_out, series, prior, sigma = layer(enc_out, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        enc_out = self.norm(enc_out)
        reconstruction = self.projection(enc_out)

        # Compute Anomaly Discrepancy Score
        discrepancy = self.compute_association_discrepancy(series_list, prior_list)

        return reconstruction, series_list, prior_list, sigma_list, discrepancy
