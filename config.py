"""Configuration dataclass for the JEPA world model pipeline."""

from dataclasses import dataclass


@dataclass
class Config:
    data_path: str = "data/pusht/pusht_cchi_v7_replay.zarr"
    subseq_len: int = 4
    history_size: int = 3
    train_split: float = 0.9
    seed: int = 42

    frameskip: int = 5

    img_size: int = 224
    vit_patch_size: int = 14
    vit_hidden_size: int = 192
    vit_num_layers: int = 12
    vit_num_heads: int = 3
    vit_mlp_ratio: float = 4.0

    latent_dim: int = 192
    action_dim: int = 2
    action_encoder_input_dim: int = 10

    predictor_type: str = "transformer"

    pred_depth: int = 6
    pred_heads: int = 16
    pred_dim_head: int = 64
    pred_mlp_dim: int = 2048
    pred_dropout: float = 0.1
    pred_emb_dropout: float = 0.0

    lstm_hidden_dim: int = 512
    lstm_num_layers: int = 2

    epochs: int = 100
    batch_size: int = 128
    lr: float = 5e-5
    weight_decay: float = 1e-3
    lambda_sigreg: float = 0.09
    grad_clip: float = 1.0
    num_workers: int = 2

    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024

    decoder_epochs: int = 30
    decoder_lr: float = 1e-3
    decoder_batch_size: int = 64
    use_lpips: bool = True

    checkpoint_dir: str = "jwm_pusht/checkpoints"
    plots_dir: str = "jwm_pusht/media"
    eval_episode_idx: int = 10

    @property
    def num_preds(self) -> int:
        return self.subseq_len - self.history_size
