"""Train the DenseNet121 character classifier."""

from __future__ import annotations

from config import DENSENET_CHECKPOINT, NUM_EPOCHS
from dataset.dataset_split import build_splits
from models.model_factory import create_model
from training.trainer import train_model


def train_densenet(epochs: int = NUM_EPOCHS) -> dict:
    splits = build_splits()
    model = create_model("densenet121", num_classes=len(splits["info"].class_names))
    history = train_model(
        model,
        splits["train_loader"],
        splits["val_loader"],
        DENSENET_CHECKPOINT,
        epochs=epochs,
    )
    return {"history": history, "splits": splits, "model": model}


if __name__ == "__main__":
    train_densenet()
