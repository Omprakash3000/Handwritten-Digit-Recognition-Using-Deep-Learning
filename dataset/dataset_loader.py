"""Load the existing Devanagari handwritten character dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    discover_dataset_path,
    ensure_output_dirs,
)
from dataset.class_labels import display_name
from utils import setup_logging

LOGGER = setup_logging(__name__)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def class_sort_key(name: str) -> tuple[int, int, str]:
    """Sort character folders before digits, using numeric class indices."""
    digit_match = re.fullmatch(r"digit_(\d+)", name)
    if digit_match:
        return (1, int(digit_match.group(1)), name)
    char_match = re.match(r"character_(\d+)_", name)
    if char_match:
        return (0, int(char_match.group(1)), name)
    return (2, 0, name)


@dataclass
class DatasetInfo:
    root: Path
    class_names: list[str]
    class_to_idx: dict[str, int]
    train_dir: Path
    test_dir: Path


def discover_classes(split_dir: Path) -> list[str]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split directory is missing: {split_dir}")
    classes = [p.name for p in split_dir.iterdir() if p.is_dir()]
    if not classes:
        raise RuntimeError(f"No class folders found in {split_dir}")
    return sorted(classes, key=class_sort_key)


def collect_samples(split_dir: Path, class_to_idx: dict[str, int]) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    missing = [name for name in class_to_idx if not (split_dir / name).is_dir()]
    if missing:
        raise RuntimeError(
            f"Dataset split {split_dir} is missing class folders: {missing[:8]}"
        )
    for class_name, idx in class_to_idx.items():
        class_dir = split_dir / class_name
        for path in class_dir.iterdir():
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, idx))
    if not samples:
        raise RuntimeError(f"No images found in {split_dir}")
    return samples


def load_dataset_info() -> DatasetInfo:
    root = discover_dataset_path()
    train_dir = root / "Train"
    test_dir = root / "Test"
    class_names = discover_classes(train_dir)
    test_classes = discover_classes(test_dir)
    if set(class_names) != set(test_classes):
        raise RuntimeError("Train and Test class folders do not match.")
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    LOGGER.info("Dataset root: %s", root)
    LOGGER.info("Detected %d classes", len(class_names))
    return DatasetInfo(
        root=root,
        class_names=class_names,
        class_to_idx=class_to_idx,
        train_dir=train_dir,
        test_dir=test_dir,
    )


class CharacterDataset(Dataset):
    """Simple image-folder dataset that keeps explicit sample paths."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_names: list[str],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if self.transform:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Failed to load image {path}: {exc}") from exc
        return image, label, str(path)

    def display_names(self) -> list[str]:
        return [display_name(name) for name in self.class_names]


def create_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    batch_size: int = BATCH_SIZE,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )


def save_class_names(class_names: list[str], path) -> None:
    from utils import save_json

    ensure_output_dirs()
    save_json({"class_names": class_names}, path)
