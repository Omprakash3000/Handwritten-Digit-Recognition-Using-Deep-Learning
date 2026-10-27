"""Train ResNet50, DenseNet121, and ViT, then search ensemble weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from config import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    DENSENET_CHECKPOINT,
    DEVICE,
    ENSEMBLE_WEIGHTS_PATH,
    HISTORY_PATH,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PLOT_DIR,
    RANDOM_SEED,
    RESNET_CHECKPOINT,
    VAL_SPLIT,
    VIT_CHECKPOINT,
    WEIGHT_DECAY,
    ensure_output_dirs,
)
from dataset.dataset_analysis import compute_dataset_stats, plot_class_distribution
from dataset.dataset_loader import save_class_names
from dataset.dataset_split import build_splits
from ensemble.weight_optimization import grid_search_weights
from evaluation.architecture_diagram import plot_architecture
from evaluation.plots import plot_training_history
from models.model_factory import create_model, load_checkpoint
from preprocessing.visualization import plot_augmentation_examples
from training.trainer import train_model
from utils import load_json, save_json, set_seed, setup_logging

LOGGER = setup_logging("train")
CHECKPOINT_MAP = {
    "resnet50": RESNET_CHECKPOINT,
    "densenet121": DENSENET_CHECKPOINT,
    "vit": VIT_CHECKPOINT,
}


@torch.no_grad()
def collect_probabilities(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs = []
    labels = []
    for images, targets, _paths in loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs.append(F.softmax(logits, dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def train_one(name: str, splits: dict, epochs: int, resume: bool = False) -> dict:
    num_classes = len(splits["info"].class_names)
    model = create_model(name, num_classes=num_classes, pretrained=True)
    history = train_model(
        model,
        splits["train_loader"],
        splits["val_loader"],
        CHECKPOINT_MAP[name],
        epochs=epochs,
        resume=resume,
    )
    if history.get("train_loss"):
        plot_training_history(name, history)
    return {"history": history, "model": model}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Handwritten Detection models.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "densenet121", "vit"],
        choices=["resnet50", "densenet121", "vit"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from existing checkpoints if available.",
    )
    args = parser.parse_args()

    set_seed()
    ensure_output_dirs()
    LOGGER.info("Device: %s", DEVICE)

    stats = compute_dataset_stats()
    if not (PLOT_DIR / "figure2_dataset_distribution.png").exists():
        plot_class_distribution(stats)
    if not (PLOT_DIR / "augmentation_examples.png").exists():
        plot_augmentation_examples()
    if not (PLOT_DIR / "figure1_system_architecture.png").exists():
        plot_architecture()

    splits = build_splits()
    save_class_names(splits["info"].class_names, CLASS_NAMES_PATH)
    save_json(
        {
            "image_size": IMAGE_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": args.epochs,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "val_split": VAL_SPLIT,
            "random_seed": RANDOM_SEED,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "models": args.models,
            "ensemble": "weighted soft voting",
        },
        HISTORY_PATH.parent / "table2_model_configuration.json",
    )

    histories = {}
    if args.resume and HISTORY_PATH.exists():
        try:
            histories = load_json(HISTORY_PATH)
        except Exception:
            histories = {}
    trained_models = {}
    for name in args.models:
        LOGGER.info("==== Training %s ====", name)
        result = train_one(name, splits, epochs=args.epochs, resume=args.resume)
        histories[name] = result["history"]
        trained_models[name] = result["model"]
        save_json(histories, HISTORY_PATH)

    save_json(histories, HISTORY_PATH)

    # Collect all available models (both newly trained and existing checkpoints)
    all_available_models = {}
    for name, path in CHECKPOINT_MAP.items():
        if name in trained_models:
            all_available_models[name] = trained_models[name]
        elif path.exists():
            LOGGER.info("Loading existing checkpoint for %s from %s", name, path)
            m = create_model(name, num_classes=len(splits["info"].class_names), pretrained=False)
            load_checkpoint(m, path)
            all_available_models[name] = m

    if len(all_available_models) >= 2:
        LOGGER.info(
            "Optimizing ensemble weights on the validation set using %d models: %s",
            len(all_available_models),
            list(all_available_models.keys()),
        )
        val_probs = {}
        val_labels = None
        for name, m in all_available_models.items():
            probs, labels = collect_probabilities(m, splits["val_loader"])
            val_probs[name] = probs
            val_labels = labels
        search = grid_search_weights(val_probs, val_labels)
        save_json(search, ENSEMBLE_WEIGHTS_PATH)
        LOGGER.info("Saved ensemble weights to %s", ENSEMBLE_WEIGHTS_PATH)
    else:
        LOGGER.info("Skipping ensemble search because fewer than 2 models are available.")

    LOGGER.info("Training complete. Checkpoints are in %s", Path(CHECKPOINT_MAP["resnet50"]).parent)


if __name__ == "__main__":
    main()
