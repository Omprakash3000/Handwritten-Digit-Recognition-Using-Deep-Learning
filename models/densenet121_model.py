"""DenseNet121 transfer-learning classifier."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121

from config import DROPOUT


def build_densenet121(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(DROPOUT),
        nn.Linear(in_features, num_classes),
    )
    return model


def densenet_target_layer(model: nn.Module) -> nn.Module:
    return model.features.denseblock4
