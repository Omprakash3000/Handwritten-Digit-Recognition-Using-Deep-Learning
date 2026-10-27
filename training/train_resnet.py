"""Train the ResNet50 character classifier."""

from __future__ import annotations

from config import NUM_EPOCHS, RESNET_CHECKPOINT
from dataset.dataset_split import build_splits
from models.model_factory import create_model
from training.trainer import train_model


def train_resnet(epochs: int = NUM_EPOCHS) -> dict:
    splits = build_splits()
    model = create_model("resnet50", num_classes=len(splits["info"].class_names))
    history = train_model(
        model,
        splits["train_loader"],
        splits["val_loader"],
        RESNET_CHECKPOINT,
        epochs=epochs,
    )
    return {"history": history, "splits": splits, "model": model}


if __name__ == "__main__":
    train_resnet()
