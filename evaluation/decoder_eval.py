"""Decoder reconstruction evaluation with 4-row comparison grid."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from jwm_pusht.models.encoder import preprocess_for_vit


def _load_episode_raw(zarr_path, ep_idx):
    root = zarr.open(zarr_path, mode="r")
    ends = root["meta"]["episode_ends"][:]
    start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    end = int(ends[ep_idx])
    imgs_np = root["data"]["img"][start:end]
    actions_np = root["data"]["action"][start:end].astype(np.float32)
    imgs_01 = torch.from_numpy(np.transpose(imgs_np.astype(np.float32), (0, 3, 1, 2))) / 255.0
    actions_t = torch.from_numpy(actions_np)
    return imgs_np, imgs_01, actions_t


@torch.no_grad()
def _decode_patches(world_model, patch_decoder, imgs_01, device, chunk=64):
    T = imgs_01.shape[0]
    out = []
    for i in range(0, T, chunk):
        batch = imgs_01[i:i + chunk].to(device).float()
        batch_pre = preprocess_for_vit(batch, world_model.img_size)
        _, patches = world_model.encoder(batch_pre, return_patch_tokens=True)
        out.append(patch_decoder(patches.to(device)).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _decode_latent(world_model, latent_decoder, imgs_01, device, chunk=64):
    recons = []
    for i in range(0, len(imgs_01), chunk):
        z = world_model.encode_single(imgs_01[i:i + chunk].to(device))
        recons.append(latent_decoder(z).cpu())
    return torch.cat(recons, dim=0)


@torch.no_grad()
def _decode_predicted(world_model, latent_decoder, imgs_01, actions_t, device, history_size=3, frameskip=5):
    H = history_size
    T = len(imgs_01)
    init_imgs = imgs_01[:H].unsqueeze(0).to(device)
    init_actions = actions_t[:H].unsqueeze(0).to(device)
    future_acts = actions_t[H:].unsqueeze(0).to(device)

    pred_emb = world_model.rollout(
        init_imgs=init_imgs, init_actions=init_actions,
        future_actions=future_acts, history_size=H, frameskip=frameskip,
    ).squeeze(0)

    z_direct = world_model.encode_single(imgs_01[:H].to(device))
    cos_sim = torch.nn.functional.cosine_similarity(z_direct, pred_emb[:H], dim=-1)
    print(f"  Rollout init frames cos_sim: {cos_sim.mean().item():.4f}")

    recons = []
    for i in range(0, len(pred_emb), 64):
        recons.append(latent_decoder(pred_emb[i:i + 64].to(device)).cpu())
    return torch.cat(recons, dim=0)


def _save_4row_grid(real_imgs, patch_dec, latent_dec, pred_dec, save_path, title, n_show=10):
    T = min(len(real_imgs), len(patch_dec), len(latent_dec), len(pred_dec))
    indices = np.linspace(0, T - 1, min(n_show, T), dtype=int)
    n = len(indices)

    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9))
    if n == 1:
        axes = axes[:, None]

    row_labels = ["Real", "PatchToken", "Latent (CLS)", "Predicted (rollout)"]
    row_data = [real_imgs, patch_dec, latent_dec, pred_dec]

    for row_idx, (label, data) in enumerate(zip(row_labels, row_data)):
        for col, t in enumerate(indices):
            axes[row_idx, col].imshow(data[t].permute(1, 2, 0).clamp(0, 1).numpy())
            if row_idx == 0:
                axes[row_idx, col].set_title(f"t={t}", fontsize=7)
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_decoder_evaluation(cfg, world_model, patch_decoder, latent_decoder, device):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    world_model.eval(); patch_decoder.eval(); latent_decoder.eval()

    _, imgs_01, actions_t = _load_episode_raw(cfg.data_path, cfg.eval_episode_idx)

    recon_patch = _decode_patches(world_model, patch_decoder, imgs_01, device)
    recon_latent = _decode_latent(world_model, latent_decoder, imgs_01, device)
    recon_pred = _decode_predicted(world_model, latent_decoder, imgs_01, actions_t, device,
                                   cfg.history_size, cfg.frameskip)

    T_min = min(len(imgs_01), len(recon_patch), len(recon_latent), len(recon_pred))
    _save_4row_grid(imgs_01[:T_min], recon_patch[:T_min], recon_latent[:T_min], recon_pred[:T_min],
                    plots_dir / "decoder_comparison.png", "Decoder Comparison")
