"""ViT-Tiny image encoder, projection MLP, and action embedder.

ViT-Tiny: hidden_size=192, depth=12, heads=3, patch_size=14, image_size=224.
Projector: Linear → BatchNorm1d → GELU → Linear (anti-collapse).
Action Embedder: Conv1d temporal smoothing + two-layer MLP with SiLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def imagenet_normalize(x):
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def imagenet_denormalize(x):
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0.0, 1.0)


def preprocess_for_vit(imgs_raw, img_size=224):
    if imgs_raw.shape[-1] != img_size or imgs_raw.shape[-2] != img_size:
        imgs_raw = F.interpolate(imgs_raw.float(), size=(img_size, img_size),
                                 mode="bilinear", align_corners=False)
    return imagenet_normalize(imgs_raw.float())


class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=14, hidden_size=192,
                 num_layers=12, num_heads=3, mlp_ratio=4.0):
        super().__init__()
        from transformers import ViTConfig, ViTModel
        config = ViTConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=int(hidden_size * mlp_ratio),
            image_size=img_size, patch_size=patch_size,
            num_channels=3, qkv_bias=True,
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        )
        self.vit = ViTModel(config, add_pooling_layer=False)
        self.hidden_size = hidden_size

    def forward(self, x, return_patch_tokens=False):
        out = self.vit(pixel_values=x, interpolate_pos_encoding=True)
        cls = out.last_hidden_state[:, 0]
        if return_patch_tokens:
            return cls, out.last_hidden_state[:, 1:]
        return cls


class ProjectionMLP(nn.Module):
    def __init__(self, dim=192, hidden_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class ActionEmbedder(nn.Module):
    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=192, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x
