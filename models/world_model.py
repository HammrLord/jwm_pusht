"""LeWorldModel: full JEPA pipeline from pixels to predicted latents.

Pipeline:
  raw pixels → ViT-Tiny → CLS → ProjectionMLP → z
  actions → ActionEmbedder → a
  z[:,:H], a[:,:H] → ARPredictor (causal Transformer + AdaLN-Zero)
                   → PredProjectionMLP → ẑ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .encoder import ActionEmbedder, ProjectionMLP, ViTEncoder, preprocess_for_vit
from .predictor import ARPredictor, LSTMPredictor


def _stack_actions(actions, frameskip):
    B, T, A = actions.shape
    padded = F.pad(actions, (0, 0, 0, frameskip - 1))
    stacked = []
    for t in range(T):
        stacked.append(padded[:, t:t + frameskip].reshape(B, 1, A * frameskip))
    return torch.cat(stacked, dim=1)


class LeWorldModel(nn.Module):
    def __init__(self, latent_dim=192, action_dim=2, action_encoder_input_dim=10,
                 img_size=224, predictor_type="transformer", vit_kwargs=None, **pred_kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        vit_kwargs = vit_kwargs or {}

        self.encoder = ViTEncoder(img_size=img_size, hidden_size=latent_dim, **vit_kwargs)
        self.projector = ProjectionMLP(dim=latent_dim, hidden_dim=2048)

        self.action_embedder = ActionEmbedder(
            input_dim=action_encoder_input_dim,
            smoothed_dim=action_encoder_input_dim,
            emb_dim=latent_dim, mlp_scale=4,
        )

        if predictor_type == "transformer":
            self.predictor = ARPredictor(latent_dim=latent_dim, **pred_kwargs)
        elif predictor_type == "lstm":
            self.predictor = LSTMPredictor(latent_dim=latent_dim, **pred_kwargs)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type!r}")

        self.pred_proj = ProjectionMLP(dim=latent_dim, hidden_dim=2048)

    def encode(self, imgs, actions, return_patch_tokens=False):
        B, T = imgs.shape[:2]
        imgs_flat = rearrange(imgs.float(), "b t c h w -> (b t) c h w")
        imgs_preprocessed = preprocess_for_vit(imgs_flat, self.img_size)

        if return_patch_tokens:
            z_flat, patches_flat = self.encoder(imgs_preprocessed, return_patch_tokens=True)
        else:
            z_flat = self.encoder(imgs_preprocessed)

        z_flat = self.projector(z_flat)
        emb = rearrange(z_flat, "(b t) d -> b t d", b=B)
        act_emb = self.action_embedder(actions)

        if return_patch_tokens:
            return emb, act_emb, patches_flat
        return emb, act_emb

    def predict(self, ctx_emb, ctx_act):
        preds = self.predictor(ctx_emb, ctx_act)
        B, H, _ = preds.shape
        preds_flat = rearrange(preds, "b t d -> (b t) d")
        preds_flat = self.pred_proj(preds_flat)
        return rearrange(preds_flat, "(b t) d -> b t d", b=B)

    def forward(self, imgs, actions, history_size=3):
        emb, act_emb = self.encode(imgs, actions)
        ctx_emb = emb[:, :history_size]
        ctx_act = act_emb[:, :history_size]
        pred_emb = self.predict(ctx_emb, ctx_act)
        tgt_emb = emb[:, 1:history_size + 1]
        return pred_emb, tgt_emb, emb

    @torch.no_grad()
    def rollout(self, init_imgs, init_actions, future_actions, history_size=3, frameskip=5):
        H = init_actions.size(1)
        device = init_imgs.device
        all_actions = torch.cat([init_actions.to(device), future_actions.to(device)], dim=1)
        all_actions_stacked = _stack_actions(all_actions, frameskip)
        init_actions_stacked = all_actions_stacked[:, :H]
        future_actions_stacked = all_actions_stacked[:, H:]

        emb, act_emb = self.encode(init_imgs, init_actions_stacked)
        future_act_emb = self.action_embedder(future_actions_stacked.float())

        for t in range(future_act_emb.size(1)):
            ctx_z = emb[:, -history_size:]
            ctx_a = act_emb[:, -history_size:]
            next_z = self.predict(ctx_z, ctx_a)[:, -1:]
            emb = torch.cat([emb, next_z], dim=1)
            act_emb = torch.cat([act_emb, future_act_emb[:, t:t + 1]], dim=1)

        return emb

    @torch.no_grad()
    def encode_single(self, img, return_patch_tokens=False):
        squeeze = img.dim() == 3
        if squeeze:
            img = img.unsqueeze(0)
        img_pre = preprocess_for_vit(img.float(), self.img_size)
        if return_patch_tokens:
            z, patches = self.encoder(img_pre, return_patch_tokens=True)
        else:
            z = self.encoder(img_pre)
        z = self.projector(z)
        if squeeze:
            z = z.squeeze(0)
            if return_patch_tokens:
                patches = patches.squeeze(0)
        if return_patch_tokens:
            return z, patches
        return z
