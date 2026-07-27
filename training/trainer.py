"""Training loop for LeWorldModel with SIGReg regularisation."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm


def build_model(cfg):
    from jwm_pusht.models.world_model import LeWorldModel

    vit_kwargs = dict(
        patch_size=cfg.vit_patch_size,
        num_layers=cfg.vit_num_layers,
        num_heads=cfg.vit_num_heads,
        mlp_ratio=cfg.vit_mlp_ratio,
    )

    pred_kwargs = {}
    if cfg.predictor_type == "transformer":
        pred_kwargs = dict(
            num_frames=cfg.history_size, depth=cfg.pred_depth,
            heads=cfg.pred_heads, dim_head=cfg.pred_dim_head,
            mlp_dim=cfg.pred_mlp_dim, dropout=cfg.pred_dropout,
            emb_dropout=cfg.pred_emb_dropout,
        )
    elif cfg.predictor_type == "lstm":
        pred_kwargs = dict(hidden_dim=cfg.lstm_hidden_dim, num_layers=cfg.lstm_num_layers)

    return LeWorldModel(
        latent_dim=cfg.latent_dim, action_dim=cfg.action_dim,
        action_encoder_input_dim=cfg.action_encoder_input_dim,
        img_size=cfg.img_size, predictor_type=cfg.predictor_type,
        vit_kwargs=vit_kwargs, **pred_kwargs,
    )


def build_sigreg(cfg):
    from jwm_pusht.models.sigreg import SIGReg
    return SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)


def train_world_model(cfg, train_loader, val_loader, device):
    model = build_model(cfg).to(device)
    sigreg = build_sigreg(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  World model parameters: {n_params:,}")

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (decay if param.dim() >= 2 else no_decay).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
    )

    warmup_epochs = 5
    schedulerwarmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cfg.epochs - warmup_epochs, eta_min=cfg.lr * 0.01)
    scheduler = SequentialLR(optimizer, schedulers=[schedulerwarmup, scheduler_cosine], milestones=[warmup_epochs])

    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_pred": [], "train_sig": [],
               "val_loss": [], "val_pred": [], "val_sig": []}
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sigreg.train()
        t_loss = t_pred = t_sig = 0.0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch:3d}/{cfg.epochs} [train]", leave=False)
        for imgs, actions in pbar:
            imgs, actions = imgs.to(device), actions.to(device)
            optimizer.zero_grad()
            pred_emb, tgt_emb, emb = model(imgs, actions, history_size=cfg.history_size)
            loss_pred = F.mse_loss(pred_emb, tgt_emb)
            loss_sig = sigreg(emb.transpose(0, 1))
            loss = loss_pred + cfg.lambda_sigreg * loss_sig
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            t_loss += loss.item()
            t_pred += loss_pred.item()
            t_sig += loss_sig.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = len(train_loader)
        t_loss /= n; t_pred /= n; t_sig /= n

        model.eval()
        v_loss = v_pred = v_sig = 0.0
        with torch.no_grad():
            for imgs, actions in val_loader:
                imgs, actions = imgs.to(device), actions.to(device)
                pred_emb, tgt_emb, emb = model(imgs, actions, history_size=cfg.history_size)
                lp = F.mse_loss(pred_emb, tgt_emb)
                ls = sigreg(emb.transpose(0, 1))
                v_loss += (lp + cfg.lambda_sigreg * ls).item()
                v_pred += lp.item()
                v_sig += ls.item()

        m = max(len(val_loader), 1)
        v_loss /= m; v_pred /= m; v_sig /= m
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["train_pred"].append(t_pred)
        history["train_sig"].append(t_sig)
        history["val_loss"].append(v_loss)
        history["val_pred"].append(v_pred)
        history["val_sig"].append(v_sig)

        print(f"  Epoch {epoch:3d}/{cfg.epochs} | "
              f"Train total={t_loss:.4f} pred={t_pred:.4f} sig={t_sig:.4f} | "
              f"Val total={v_loss:.4f} pred={v_pred:.4f} sig={v_sig:.4f}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), ckpt_dir / "world_model_best.pt")

    torch.save(model.state_dict(), ckpt_dir / "world_model_final.pt")
    return history, model
