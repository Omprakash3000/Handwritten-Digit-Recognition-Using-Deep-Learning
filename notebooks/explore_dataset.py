"""Lightweight dataset inspection notebook-style script."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.dataset_analysis import compute_dataset_stats, dataset_table, plot_class_distribution
from evaluation.architecture_diagram import plot_architecture
from preprocessing.visualization import plot_augmentation_examples


def main() -> None:
    stats = compute_dataset_stats()
    print("Dataset root:", stats["dataset_root"])
    print("Classes:", stats["num_classes"])
    print("Train / val / test:", stats["training_images"], stats["validation_images"], stats["test_images"])
    print(dataset_table(stats).head())
    print("Distribution plot:", plot_class_distribution(stats))
    print("Augmentation plot:", plot_augmentation_examples())
    print("Architecture plot:", plot_architecture())


if __name__ == "__main__":
    main()
