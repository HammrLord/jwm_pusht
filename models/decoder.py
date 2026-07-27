"""Two decoders for latent-to-pixel reconstruction.

PatchTokenDecoder: decodes ViT patch tokens (16×16 grid × 192D) via PixelShuffle.
LatentDecoder: decodes a 192D CLS vector via progressive upsampling with residual blocks.
LPIPSLoss: perceptual loss using AlexNet (Zhang et al., 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = _conv3x3(in_ch, out_ch) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _conv3x3(in_ch, out_ch)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(self.up(x))))


class PatchTokenDecoder(nn.Module):
    def __init__(self, embed_dim=192, num_patches=256, out_size=96):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = int(num_patches ** 0.5)
        self.out_size = out_size
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.stage2 = nn.Sequential(ResBlock(48, 64), UpsampleBlock(64, 64))
        self.stage3 = nn.Sequential(ResBlock(64, 32), UpsampleBlock(32, 16), _conv3x3(16, 3))

    def forward(self, patch_tokens):
        B = patch_tokens.size(0)
        x = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
        x = self.pixel_shuffle(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class LatentDecoder(nn.Module):
    def __init__(self, latent_dim=192, out_size=96):
        super().__init__()
        self.out_size = out_size
        base_ch = 1024
        self.expand = nn.Sequential(
            nn.Linear(latent_dim, base_ch * 4 * 4),
            nn.Unflatten(1, (base_ch, 4, 4)),
        )
        self.blocks = nn.Sequential(
            ResBlock(1024, 512), UpsampleBlock(512, 256),
            ResBlock(256, 256),  UpsampleBlock(256, 128),
            ResBlock(128, 128),  UpsampleBlock(128, 64),
            ResBlock(64, 64),    UpsampleBlock(64, 32),
            ResBlock(32, 32),    _conv3x3(32, 3),
        )

    def forward(self, z):
        x = self.expand(z)
        x = self.blocks(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size),
                              mode="bilinear", align_corners=False)
        return torch.sigmoid(x)


class LPIPSLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._lpips = None

    def _ensure_loaded(self, device):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex", spatial=True)
            self.add_module("_lpips", self._lpips)
        self._lpips.to(device)

    def forward(self, pred, target, chunk_size=32):
        self._ensure_loaded(pred.device)
        losses = []
        for i in range(0, pred.size(0), chunk_size):
            p = pred[i:i + chunk_size]
            t = target[i:i + chunk_size]
            losses.append(self._lpips(p, t, normalize=True).mean())
        return torch.stack(losses).mean()
