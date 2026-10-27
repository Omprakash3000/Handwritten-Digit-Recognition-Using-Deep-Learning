"""Training curve plots used during and after model fitting."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from config import PLOT_DIR, ensure_output_dirs


def plot_training_history(model_name: str, history: dict) -> dict[str, Path]:
    ensure_output_dirs()
    paths = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Training Loss")
    ax.plot(history["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    loss_path = PLOT_DIR / f"{model_name}_loss.png"
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)
    paths["loss"] = loss_path

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_accuracy"], label="Training Accuracy")
    ax.plot(history["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name} — Training vs Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    acc_path = PLOT_DIR / f"{model_name}_accuracy.png"
    fig.savefig(acc_path, dpi=160)
    plt.close(fig)
    paths["accuracy"] = acc_path

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["learning_rate"], label="Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title(f"{model_name} — Learning Rate Curve")
    ax.legend()
    fig.tight_layout()
    lr_path = PLOT_DIR / f"{model_name}_learning_rate.png"
    fig.savefig(lr_path, dpi=160)
    plt.close(fig)
    paths["learning_rate"] = lr_path
    return {key: str(value) for key, value in paths.items()}
