"""Full pipeline: train world model, evaluate latents, train decoders, evaluate reconstructions.

Usage:  python3 jwm_pusht/run.py
"""

import sys, os, time
from pathlib import Path
import torch

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from jwm_pusht.config import Config
from jwm_pusht.data.dataset import get_dataloaders
from jwm_pusht.training.trainer import build_model, train_world_model
from jwm_pusht.evaluation.latent_eval import run_latent_evaluation
from jwm_pusht.evaluation.decoder_eval import run_decoder_evaluation
from jwm_pusht.training.decoder_trainer import train_decoder
from jwm_pusht.utils.plotting import plot_world_model_training, plot_decoder_training


def main():
    cfg = Config()
    t0 = time.time()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}  |  Predictor: {cfg.predictor_type}  |  Latent dim: {cfg.latent_dim}")

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.plots_dir).mkdir(parents=True, exist_ok=True)

    print("► Loading data...")
    train_loader, val_loader, n_train, n_val = get_dataloaders(cfg)
    print(f"  {n_train} train episodes, {n_val} val episodes")

    print(f"► Phase 1 — Training world model ({cfg.epochs} epochs)...")
    wm_history, world_model = train_world_model(cfg, train_loader, val_loader, device)
    plot_world_model_training(wm_history, Path(cfg.plots_dir) / "world_model_training.png")

    best_ckpt = Path(cfg.checkpoint_dir) / "world_model_best.pt"
    world_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    world_model = world_model.to(device).eval()

    print("► Phase 2 — Latent space evaluation...")
    run_latent_evaluation(cfg, world_model, device)

    print(f"► Phase 3 — Training decoders ({cfg.decoder_epochs} epochs)...")
    from jwm_pusht.data.dataset import PushTSubTrajectoryDataset
    from torch.utils.data import DataLoader
    import zarr, random

    root = zarr.open(cfg.data_path, mode="r")
    n_eps = len(root["meta"]["episode_ends"][:])
    rng = random.Random(cfg.seed)
    all_eps = list(range(n_eps)); rng.shuffle(all_eps)
    train_eps = all_eps[:int(cfg.train_split * n_eps)]

    decoder_ds = PushTSubTrajectoryDataset(cfg.data_path, cfg.subseq_len, frameskip=cfg.frameskip, episode_indices=train_eps)
    decoder_loader = DataLoader(decoder_ds, batch_size=cfg.decoder_batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    dec_history, patch_decoder, latent_decoder = train_decoder(cfg, world_model, decoder_loader, device)
    plot_decoder_training(dec_history, Path(cfg.plots_dir) / "decoder_training.png")

    from jwm_pusht.models.decoder import PatchTokenDecoder, LatentDecoder
    patch_decoder = PatchTokenDecoder(embed_dim=cfg.latent_dim, out_size=96).to(device)
    patch_decoder.load_state_dict(torch.load(Path(cfg.checkpoint_dir) / "patch_decoder_best.pt", map_location=device))
    patch_decoder.eval()

    latent_decoder = LatentDecoder(latent_dim=cfg.latent_dim, out_size=96).to(device)
    latent_decoder.load_state_dict(torch.load(Path(cfg.checkpoint_dir) / "latent_decoder_best.pt", map_location=device))
    latent_decoder.eval()

    print("► Phase 4 — Decoder evaluation...")
    run_decoder_evaluation(cfg, world_model, patch_decoder, latent_decoder, device)

    elapsed = time.time() - t0
    print(f"\nDone in {int(elapsed // 60)}m {int(elapsed % 60)}s")


if __name__ == "__main__":
    main()
