"""Training loops for both decoders."""

from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm


def _sobel(x):
    kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=x.device)
    ky = kx.T
    kx = kx.view(1, 1, 3, 3).expand(3, -1, -1, -1)
    ky = ky.view(1, 1, 3, 3).expand(3, -1, -1, -1)
    ex = F.conv2d(x, kx, padding=1, groups=3)
    ey = F.conv2d(x, ky, padding=1, groups=3)
    return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)


def decoder_loss(pred, target, lpips_loss=None):
    mse = F.mse_loss(pred, target)
    edge = F.mse_loss(_sobel(pred), _sobel(target))
    loss = mse + 0.5 * edge
    if lpips_loss is not None:
        loss = loss + 0.1 * lpips_loss(pred, target)
    return loss


def _get_patch_tokens(world_model, imgs_flat, device):
    from jwm_pusht.models.encoder import preprocess_for_vit
    imgs_pre = preprocess_for_vit(imgs_flat.to(device).float(), world_model.img_size)
    _, patches = world_model.encoder(imgs_pre, return_patch_tokens=True)
    return patches


def train_patch_decoder(cfg, world_model, data_loader, device):
    from jwm_pusht.models.decoder import PatchTokenDecoder, LPIPSLoss

    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    decoder = PatchTokenDecoder(embed_dim=cfg.latent_dim, out_size=96).to(device)
    lpips = LPIPSLoss().to(device) if getattr(cfg, 'use_lpips', True) else None
    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"loss": []}
    best_loss = float("inf")

    for epoch in range(1, cfg.decoder_epochs + 1):
        decoder.train()
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(data_loader, desc=f"  PatchDecoder epoch {epoch:3d}/{cfg.decoder_epochs}", leave=False)
        for imgs, _ in pbar:
            B, T, C, H, W = imgs.shape
            imgs_flat = imgs.view(B * T, C, H, W).to(device)
            with torch.no_grad():
                patches = _get_patch_tokens(world_model, imgs_flat, device)
            recon = decoder(patches)
            loss = decoder_loss(recon, imgs_flat, lpips)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg)
        print(f"  PatchDecoder epoch {epoch:3d}/{cfg.decoder_epochs} | loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(decoder.state_dict(), ckpt_dir / "patch_decoder_best.pt")

    torch.save(decoder.state_dict(), ckpt_dir / "patch_decoder_final.pt")
    return history, decoder


def train_latent_decoder(cfg, world_model, data_loader, device):
    from jwm_pusht.models.decoder import LatentDecoder, LPIPSLoss

    world_model.eval()
    for p in world_model.parameters():
        p.requires_grad_(False)

    decoder = LatentDecoder(latent_dim=cfg.latent_dim, out_size=96).to(device)
    lpips = LPIPSLoss().to(device) if getattr(cfg, 'use_lpips', True) else None
    optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.decoder_lr)
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"loss": []}
    best_loss = float("inf")

    for epoch in range(1, cfg.decoder_epochs + 1):
        decoder.train()
        epoch_loss, n_batches = 0.0, 0

        pbar = tqdm(data_loader, desc=f"  LatentDecoder epoch {epoch:3d}/{cfg.decoder_epochs}", leave=False)
        for imgs, _ in pbar:
            B, T, C, H, W = imgs.shape
            imgs_flat = imgs.view(B * T, C, H, W).to(device)
            with torch.no_grad():
                z_flat = world_model.encode_single(imgs_flat)
            recon = decoder(z_flat)
            loss = decoder_loss(recon, imgs_flat, lpips)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg)
        print(f"  LatentDecoder epoch {epoch:3d}/{cfg.decoder_epochs} | loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save(decoder.state_dict(), ckpt_dir / "latent_decoder_best.pt")

    torch.save(decoder.state_dict(), ckpt_dir / "latent_decoder_final.pt")
    return history, decoder


def train_decoder(cfg, world_model, data_loader, device):
    print("\n► Training PatchTokenDecoder...")
    patch_history, patch_decoder = train_patch_decoder(cfg, world_model, data_loader, device)
    print("\n► Training LatentDecoder...")
    latent_history, latent_decoder = train_latent_decoder(cfg, world_model, data_loader, device)
    return {"patch": patch_history, "latent": latent_history}, patch_decoder, latent_decoder
