"""Vision Transformer classifier based on a pretrained ViT-B/16 backbone."""

from __future__ import annotations

import timm
from torch import nn

from config import DROPOUT


def build_vit(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=DROPOUT,
    )
    return model


def vit_patch_size(model: nn.Module) -> int:
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None and hasattr(patch_embed, "patch_size"):
        size = patch_embed.patch_size
        if isinstance(size, tuple):
            return int(size[0])
        return int(size)
    return 16
