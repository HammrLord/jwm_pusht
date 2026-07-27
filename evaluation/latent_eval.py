"""Latent space evaluation: PCA trajectory plots and distribution histograms."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_episode(zarr_path, ep_idx):
    root = zarr.open(zarr_path, mode="r")
    ends = root["meta"]["episode_ends"][:]
    start = 0 if ep_idx == 0 else int(ends[ep_idx - 1])
    end = int(ends[ep_idx])
    imgs = root["data"]["img"][start:end]
    actions = root["data"]["action"][start:end]
    imgs = np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255.0
    return torch.from_numpy(imgs), torch.from_numpy(actions.astype(np.float32))


@torch.no_grad()
def encode_episode(model, imgs_t, device):
    T, chunk = imgs_t.shape[0], 64
    zs = []
    for i in range(0, T, chunk):
        z = model.encode_single(imgs_t[i:i+chunk].to(device))
        zs.append(z.cpu().numpy())
    return np.concatenate(zs, axis=0)


@torch.no_grad()
def open_loop_rollout(model, imgs_t, actions_t, device, history_size=3):
    T = imgs_t.shape[0]
    imgs_t, actions_t = imgs_t.to(device), actions_t.to(device)
    all_emb = model.rollout(
        init_imgs=imgs_t[:history_size].unsqueeze(0),
        init_actions=actions_t[:history_size].unsqueeze(0),
        future_actions=actions_t[history_size:].unsqueeze(0),
        history_size=history_size,
    )
    return all_emb.squeeze(0).cpu().numpy()


def plot_latent_trajectory(z_real, z_pred, save_path):
    T = min(len(z_real), len(z_pred))
    z_real, z_pred = z_real[:T], z_pred[:T]

    pca = PCA(n_components=2)
    pca.fit(np.concatenate([z_real, z_pred], axis=0))
    real_2d = pca.transform(z_real)
    pred_2d = pca.transform(z_pred)
    mae_2d = np.mean(np.abs(real_2d - pred_2d))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(real_2d[:, 0], real_2d[:, 1], "b-o", label="Real", markersize=3, alpha=0.7)
    ax.plot(pred_2d[:, 0], pred_2d[:, 1], "r--o", label="Predicted (open-loop)", markersize=3, alpha=0.7)
    ax.scatter(*real_2d[0], c="green", s=150, zorder=5, marker="*", label="Start")
    ax.scatter(*real_2d[-1], c="black", s=100, zorder=5, marker="s", label="End (real)")
    ax.scatter(*pred_2d[-1], c="darkred", s=100, zorder=5, marker="X", label="End (pred)")
    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% var)")
    ax.set_title(f"Latent Trajectory — Real vs Open-Loop Prediction\nMAE in PCA space: {mae_2d:.4f}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    return mae_2d


def plot_latent_distributions(z_real, save_path, n_dims=8):
    scaler = StandardScaler()
    z_std = scaler.fit_transform(z_real)
    n_dims = min(n_dims, z_real.shape[1])
    ncols, nrows = 4, (n_dims + 3) // 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()
    x_ref = np.linspace(-4, 4, 200)
    ref_pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_ref ** 2)
    for i in range(n_dims):
        axes[i].hist(z_std[:, i], bins=50, density=True, alpha=0.6, color="steelblue")
        axes[i].plot(x_ref, ref_pdf, "r-", linewidth=1.5)
        axes[i].set_title(f"Dim {i}", fontsize=9)
        axes[i].set_xlim(-4, 4)
    for j in range(n_dims, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Latent Dimension Distributions vs N(0,1)", fontsize=12)
    fig.tight_layout(); save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)


def run_latent_evaluation(cfg, model, device):
    plots_dir = Path(cfg.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    imgs_t, actions_t = load_episode(cfg.data_path, cfg.eval_episode_idx)
    z_real = encode_episode(model, imgs_t, device)
    z_pred = open_loop_rollout(model, imgs_t, actions_t, device, cfg.history_size)
    T = min(len(z_real), len(z_pred))
    mae = plot_latent_trajectory(z_real[:T], z_pred[:T], plots_dir / "latent_trajectory.png")
    plot_latent_distributions(z_real, plots_dir / "latent_distributions.png")
    return mae
