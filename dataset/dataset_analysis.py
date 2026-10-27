"""Dataset statistics, class distribution plots, and sample galleries."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import PLOT_DIR, RESULTS_DIR, ensure_output_dirs
from dataset.class_labels import display_name, glyph_only
from dataset.dataset_loader import collect_samples, load_dataset_info
from dataset.dataset_split import stratified_split
from utils import save_json, setup_logging

LOGGER = setup_logging(__name__)


def compute_dataset_stats(force: bool = False) -> dict:
    stats_file = RESULTS_DIR / "dataset_stats.json"
    if not force and stats_file.exists():
        try:
            from utils import load_json
            return load_json(stats_file)
        except Exception:
            pass
    info = load_dataset_info()
    train_pool = collect_samples(info.train_dir, info.class_to_idx)
    test_samples = collect_samples(info.test_dir, info.class_to_idx)
    train_samples, val_samples = stratified_split(train_pool)

    train_counts = Counter(label for _, label in train_samples)
    val_counts = Counter(label for _, label in val_samples)
    test_counts = Counter(label for _, label in test_samples)
    original_train_counts = Counter(label for _, label in train_pool)

    per_class = []
    for idx, name in enumerate(info.class_names):
        per_class.append(
            {
                "class_index": idx,
                "class_name": name,
                "display_name": display_name(name),
                "glyph": glyph_only(name),
                "original_train": original_train_counts[idx],
                "train": train_counts[idx],
                "validation": val_counts[idx],
                "test": test_counts[idx],
                "total": original_train_counts[idx] + test_counts[idx],
            }
        )

    stats = {
        "dataset_root": str(info.root),
        "num_classes": len(info.class_names),
        "class_names": info.class_names,
        "total_images": len(train_pool) + len(test_samples),
        "original_train_images": len(train_pool),
        "training_images": len(train_samples),
        "validation_images": len(val_samples),
        "test_images": len(test_samples),
        "per_class": per_class,
    }
    ensure_output_dirs()
    save_json(stats, RESULTS_DIR / "dataset_stats.json")
    LOGGER.info("Saved dataset statistics to %s", RESULTS_DIR / "dataset_stats.json")
    return stats


def plot_class_distribution(stats: dict | None = None, save_path: Path | None = None) -> Path:
    stats = stats or compute_dataset_stats()
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / "figure2_dataset_distribution.png")

    names = [row["glyph"] for row in stats["per_class"]]
    train_vals = [row["train"] for row in stats["per_class"]]
    val_vals = [row["validation"] for row in stats["per_class"]]
    test_vals = [row["test"] for row in stats["per_class"]]

    fig, ax = plt.subplots(figsize=(18, 7))
    x = range(len(names))
    ax.bar(x, train_vals, label="Train (85%)")
    ax.bar(x, val_vals, bottom=train_vals, label="Validation (15%)")
    ax.bar(
        x,
        test_vals,
        bottom=[t + v for t, v in zip(train_vals, val_vals)],
        label="Test",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=90)
    ax.set_title("Figure 2. Dataset Distribution by Character Class")
    ax.set_xlabel("Character")
    ax.set_ylabel("Image count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def dataset_table(stats: dict | None = None) -> pd.DataFrame:
    stats = stats or compute_dataset_stats()
    return pd.DataFrame(stats["per_class"])


def sample_images_for_class(class_name: str, split: str = "Train", limit: int = 8) -> list[Image.Image]:
    info = load_dataset_info()
    if class_name not in info.class_to_idx:
        raise ValueError(f"Unknown class: {class_name}")
    folder = (info.train_dir if split.lower() == "train" else info.test_dir) / class_name
    images: list[Image.Image] = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
        if len(images) >= limit:
            break
    return images
