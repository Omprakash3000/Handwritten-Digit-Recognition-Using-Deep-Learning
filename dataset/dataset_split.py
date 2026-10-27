"""Stratified train/validation split on the original Train folder."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from config import RANDOM_SEED, VAL_SPLIT
from dataset.dataset_loader import (
    CharacterDataset,
    DatasetInfo,
    collect_samples,
    create_dataloader,
    load_dataset_info,
)
from preprocessing.transforms import eval_transforms, train_transforms
from utils import setup_logging

LOGGER = setup_logging(__name__)


def stratified_split(
    samples: list[tuple[Path, int]],
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    labels = [label for _, label in samples]
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )
    return train_samples, val_samples


def build_splits(
    info: DatasetInfo | None = None,
) -> dict:
    """Create dataloaders while keeping the original Test split untouched."""
    info = info or load_dataset_info()
    train_pool = collect_samples(info.train_dir, info.class_to_idx)
    test_samples = collect_samples(info.test_dir, info.class_to_idx)
    train_samples, val_samples = stratified_split(train_pool)

    train_ds = CharacterDataset(train_samples, info.class_names, train_transforms())
    val_ds = CharacterDataset(val_samples, info.class_names, eval_transforms())
    test_ds = CharacterDataset(test_samples, info.class_names, eval_transforms())

    LOGGER.info(
        "Split sizes | train=%d val=%d test=%d classes=%d",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        len(info.class_names),
    )
    return {
        "info": info,
        "train_dataset": train_ds,
        "val_dataset": val_ds,
        "test_dataset": test_ds,
        "train_loader": create_dataloader(train_ds, shuffle=True),
        "val_loader": create_dataloader(val_ds, shuffle=False),
        "test_loader": create_dataloader(test_ds, shuffle=False),
        "counts": {
            "total": len(train_pool) + len(test_samples),
            "train": len(train_samples),
            "validation": len(val_samples),
            "test": len(test_samples),
            "num_classes": len(info.class_names),
            "train_per_class": dict(Counter(label for _, label in train_samples)),
            "val_per_class": dict(Counter(label for _, label in val_samples)),
            "test_per_class": dict(Counter(label for _, label in test_samples)),
        },
    }
