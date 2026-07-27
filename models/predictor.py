"""Autoregressive predictors: causal Transformer with AdaLN-Zero (paper) and LSTM baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) \
            if inner_dim != dim else nn.Linear(inner_dim, dim)

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=True)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class AdaLNBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = CausalSelfAttention(dim, heads, dim_head, dropout)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_a * self.attn(modulate(self.norm1(x), shift_a, scale_a))
        x = x + gate_f * self.ffn(modulate(self.norm2(x), shift_f, scale_f))
        return x


class ARPredictor(nn.Module):
    def __init__(self, latent_dim=192, num_frames=3, depth=4, heads=4,
                 dim_head=48, mlp_dim=512, dropout=0.1, emb_dropout=0.0):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, latent_dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            AdaLNBlock(latent_dim, heads, dim_head, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z_seq, a_seq):
        T = z_seq.size(1)
        x = z_seq + self.pos_embedding[:, :T]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, a_seq)
        return self.norm(x)


class LSTMPredictor(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=latent_dim * 2, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_seq, a_seq):
        x = torch.cat([z_seq, a_seq], dim=-1)
        out, _ = self.lstm(x)
        return self.out_proj(out)
