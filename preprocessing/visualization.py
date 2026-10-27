"""Visualize original vs augmented handwritten characters."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from config import PLOT_DIR, ensure_output_dirs
from dataset.dataset_loader import collect_samples, load_dataset_info
from preprocessing.augmentation import preview_augmentation
from preprocessing.transforms import visualization_transforms


def plot_augmentation_examples(num_images: int = 6, save_path: Path | None = None) -> Path:
    info = load_dataset_info()
    samples = collect_samples(info.train_dir, info.class_to_idx)[:num_images]
    original_tf = visualization_transforms()
    aug_tf = preview_augmentation()

    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / "augmentation_examples.png")
    fig, axes = plt.subplots(num_images, 2, figsize=(6, 3 * num_images))
    if num_images == 1:
        axes = [axes]
    for row, (path, _) in enumerate(samples):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            original = original_tf(rgb)
            augmented = aug_tf(rgb)
        axes[row][0].imshow(original)
        axes[row][0].set_title("Original Image")
        axes[row][0].axis("off")
        axes[row][1].imshow(augmented)
        axes[row][1].set_title("Augmented Image")
        axes[row][1].axis("off")
        axes[row][0].annotate(
            "",
            xy=(1.05, 0.5),
            xycoords="axes fraction",
            xytext=(1.0, 0.5),
            arrowprops={"arrowstyle": "->"},
        )
    fig.suptitle("Original → Augmented Character Samples", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return save_path
