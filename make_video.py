"""Render full-episode side-by-side comparison videos (real vs JEPA predicted).

Usage:  python3 make_video.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, zarr, numpy as np, random, cv2
from jwm_pusht.config import Config
from jwm_pusht.training.trainer import build_model
from jwm_pusht.models.decoder import LatentDecoder


def remap_vit_keys(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k.replace("encoder.vit.layers.", "encoder.vit.encoder.layer.")
        nk = nk.replace("attention.q_proj.", "attention.attention.query.")
        nk = nk.replace("attention.k_proj.", "attention.attention.key.")
        nk = nk.replace("attention.v_proj.", "attention.attention.value.")
        nk = nk.replace("attention.o_proj.", "attention.output.dense.")
        nk = nk.replace("mlp.fc1.", "intermediate.dense.")
        nk = nk.replace("mlp.fc2.", "output.dense.")
        out[nk] = v
    return out


def decode_batch(ldec, latents, device):
    results = []
    for i in range(0, len(latents), 4):
        results.append(ldec(latents[i:i+4].to(device)).cpu())
    return torch.cat(results)


def img_to_frame(tensor):
    return (tensor.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def make_episode_video(model, ldec, ep, s, e, root, device, H, frameskip, plots_dir):
    T = e - s
    imgs = torch.from_numpy(np.transpose(root["data"]["img"][s:e].astype(np.float32), (0, 3, 1, 2))) / 255.0
    acts = torch.from_numpy(root["data"]["action"][s:e].astype(np.float32))

    with torch.no_grad():
        pred_latents = model.rollout(
            imgs[:H].unsqueeze(0).to(device), acts[:H].unsqueeze(0).to(device),
            acts[H:].unsqueeze(0).to(device), H, frameskip,
        ).squeeze(0)
        pred_frames = [decode_batch(ldec, pred_latents[t:t+1], device).squeeze(0) for t in range(T)]

    cell, gap = 384, 10
    width, height = 2 * cell + gap, cell + 80
    out_path = Path(plots_dir) / f"episode_{ep}.mp4"

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for t in range(T):
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 240
        real = cv2.resize(img_to_frame(imgs[t]), (cell, cell), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(img_to_frame(pred_frames[t]), (cell, cell), interpolation=cv2.INTER_NEAREST)
        canvas[50:50+cell, 0:cell] = real
        canvas[50:50+cell, cell+gap:2*cell+gap] = pred

        in_init = t < H
        c = (255, 255, 255) if in_init else (0, 0, 0)
        cv2.putText(canvas, "Real", (cell//2 - 20, 35), font, 0.7, c, 2)
        cv2.putText(canvas, "Predicted (JEPA)", (cell + gap + cell//2 - 80, 35), font, 0.7, c, 2)
        phase = "INIT (GT context)" if in_init else "OPEN-LOOP PREDICTION"
        color = (200, 200, 200) if in_init else (0, 200, 0)
        cv2.putText(canvas, f"Ep {ep}  t={t}/{T}  {phase}", (10, height - 15), font, 0.5, color, 1)
        if not in_init:
            cv2.rectangle(canvas, (0, 0), (width-1, height-1), (0, 220, 0), 4)
        writer.write(canvas)

    writer.release()
    print(f"  Episode {ep}: {T} frames → {out_path}")


def main():
    cfg, device = Config(), torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    H = cfg.history_size

    project_root = Path(__file__).resolve().parent
    ckpt_dir = project_root / "checkpoints"
    media_dir = project_root / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    data_path = project_root.parent / "data" / "pusht" / "pusht_cchi_v7_replay.zarr"

    print("Loading models...")
    model = build_model(cfg)
    raw = torch.load(ckpt_dir / "world_model_best.pt", map_location="cpu")
    model.load_state_dict(remap_vit_keys(raw), strict=False)
    model = model.to(device).eval()

    ldec = LatentDecoder(192, 96).to(device)
    ldec.load_state_dict(torch.load(ckpt_dir / "latent_decoder_best.pt", map_location=device))
    ldec.eval()

    root = zarr.open(str(data_path), mode="r")
    ends = root["meta"]["episode_ends"][:]
    total_eps = len(ends)

    random.seed(123)
    candidates = [e for e in range(total_eps) if e != 10 and (int(ends[e]) - (0 if e == 0 else int(ends[e-1]))) >= 50]
    eps = [10] + random.sample(candidates, 3)

    print(f"Episodes: {eps}")
    for ep in eps:
        s = 0 if ep == 0 else int(ends[ep - 1])
        e = int(ends[ep])
        make_episode_video(model, ldec, ep, s, e, root, device, H, cfg.frameskip, media_dir)

    print(f"\nDone. Videos in {media_dir}/")


if __name__ == "__main__":
    main()
