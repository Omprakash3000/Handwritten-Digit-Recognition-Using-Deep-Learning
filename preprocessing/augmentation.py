"""Conservative augmentations that preserve Devanagari character identity."""

from __future__ import annotations

from torchvision import transforms

from config import IMAGE_SIZE


def conservative_train_augmentation() -> list:
    """
    Horizontal flips are omitted because they change Devanagari identity.
    Rotations, crops, and photometric jitter stay small for the same reason.
    """
    return [
        transforms.RandomRotation(degrees=8, fill=255),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.94, 1.06),
            fill=255,
        ),
        transforms.RandomResizedCrop(
            IMAGE_SIZE,
            scale=(0.88, 1.0),
            ratio=(0.95, 1.05),
        ),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]


def preview_augmentation() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            *conservative_train_augmentation(),
        ]
    )
