"""PushT sub-trajectory dataset from zarr replay buffer.

Each sample: `subseq_len` contiguous frames with frameskip-stacked actions.
"""

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class PushTSubTrajectoryDataset(Dataset):
    def __init__(self, zarr_path, subseq_len=4, frameskip=5, episode_indices=None):
        super().__init__()
        self.root = zarr.open(zarr_path, mode="r")
        self.episode_ends = self.root["meta"]["episode_ends"][:]
        self.subseq_len = subseq_len
        self.frameskip = frameskip

        if episode_indices is None:
            episode_indices = list(range(len(self.episode_ends)))

        self.valid_starts = []
        for i in episode_indices:
            ep_start = 0 if i == 0 else int(self.episode_ends[i - 1])
            ep_end = int(self.episode_ends[i])
            if ep_end - ep_start >= subseq_len + frameskip - 1:
                self.valid_starts.extend(range(ep_start, ep_end - subseq_len - frameskip + 2))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end = start + self.subseq_len

        imgs = self.root["data"]["img"][start:end]
        actions = self.root["data"]["action"][start:end]

        if self.frameskip > 1:
            stacked_actions = []
            for t in range(self.subseq_len):
                a_start = start + t
                a_end = a_start + self.frameskip
                stacked_actions.append(self.root["data"]["action"][a_start:a_end].flatten())
            actions = np.array(stacked_actions)

        imgs = np.transpose(imgs, (0, 3, 1, 2))
        imgs_t = torch.from_numpy(imgs.copy()).float().div_(255.0)
        actions_t = torch.from_numpy(actions.copy()).float()
        return imgs_t, actions_t


def get_dataloaders(cfg):
    import random
    from torch.utils.data import DataLoader

    root = zarr.open(cfg.data_path, mode="r")
    num_episodes = len(root["meta"]["episode_ends"][:])

    rng = random.Random(cfg.seed)
    all_eps = list(range(num_episodes))
    rng.shuffle(all_eps)

    n_train = int(cfg.train_split * num_episodes)
    train_eps = all_eps[:n_train]
    val_eps = all_eps[n_train:]

    train_ds = PushTSubTrajectoryDataset(cfg.data_path, cfg.subseq_len, frameskip=cfg.frameskip, episode_indices=train_eps)
    val_ds = PushTSubTrajectoryDataset(cfg.data_path, cfg.subseq_len, frameskip=cfg.frameskip, episode_indices=val_eps)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader, len(train_eps), len(val_eps)
