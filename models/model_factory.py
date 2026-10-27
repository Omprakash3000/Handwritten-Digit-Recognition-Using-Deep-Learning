"""Create, freeze/unfreeze, and load classification models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from config import DEVICE
from models.densenet121_model import build_densenet121
from models.resnet50_model import build_resnet50
from models.vit_model import build_vit


MODEL_BUILDERS = {
    "resnet50": build_resnet50,
    "densenet121": build_densenet121,
    "vit": build_vit,
}


def create_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    key = name.lower()
    if key not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_BUILDERS)}")
    model = MODEL_BUILDERS[key](num_classes=num_classes, pretrained=pretrained)
    return model.to(DEVICE)


def load_checkpoint(model: nn.Module, path: Path, map_location=None) -> nn.Module:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location=map_location or DEVICE, weights_only=False)
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model
