"""Training curve plots for world model and decoders."""

from pathlib import Path
import matplotlib.pyplot as plt


def plot_world_model_training(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], "b-", label="Train total")
    axes[0].plot(epochs, history["val_loss"], "r--", label="Val total")
    axes[0].set_title("Total Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_pred"], "b-", label="Train pred")
    axes[1].plot(epochs, history["val_pred"], "r--", label="Val pred")
    axes[1].set_title("Prediction Loss (MSE)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_sig"], "b-", label="Train SIGReg")
    axes[2].plot(epochs, history["val_sig"], "r--", label="Val SIGReg")
    axes[2].set_title("SIGReg Loss"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    fig.suptitle("LeWorldModel Training Curves", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_decoder_training(history, save_path):
    patch_epochs = range(1, len(history["patch"]["loss"]) + 1)
    latent_epochs = range(1, len(history["latent"]["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(patch_epochs, history["patch"]["loss"], "b-", linewidth=1.8)
    axes[0].set_title("PatchTokenDecoder"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].grid(True, alpha=0.3)

    axes[1].plot(latent_epochs, history["latent"]["loss"], "r-", linewidth=1.8)
    axes[1].set_title("LatentDecoder"); axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss"); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Decoder Training Curves", fontsize=13)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
