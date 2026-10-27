"""Collect misclassified test images and common confusion pairs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from config import PLOT_DIR, ensure_output_dirs
from dataset.class_labels import glyph_only


def collect_misclassified(
    paths: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    limit: int = 16,
) -> list[dict]:
    errors = []
    for idx, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == pred:
            continue
        errors.append(
            {
                "path": paths[idx],
                "true_class": class_names[int(true)],
                "predicted_class": class_names[int(pred)],
                "true_glyph": glyph_only(class_names[int(true)]),
                "predicted_glyph": glyph_only(class_names[int(pred)]),
                "confidence": float(y_prob[idx, int(pred)]),
            }
        )
        if len(errors) >= limit:
            break
    return errors


def plot_misclassified(errors: list[dict], save_path: Path | None = None) -> Path | None:
    if not errors:
        return None
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / "misclassified_examples.png")
    cols = 4
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, item in zip(axes, errors):
        with Image.open(item["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"True: {item['true_glyph']}\nPred: {item['predicted_glyph']}\n"
            f"{item['confidence']:.2%}",
            fontsize=9,
        )
        ax.axis("off")
    for ax in axes[len(errors) :]:
        ax.axis("off")
    fig.suptitle("Misclassified Test Characters")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path
