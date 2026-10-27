"""Confusion matrix visualization and confused-class analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from config import CM_DIR, ensure_output_dirs
from dataset.class_labels import glyph_only


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path | None = None,
) -> tuple[np.ndarray, Path]:
    ensure_output_dirs()
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    save_path = save_path or (CM_DIR / f"{title.replace(' ', '_').lower()}.png")
    labels = [glyph_only(name) for name in class_names]
    fig, ax = plt.subplots(figsize=(16, 14))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    return matrix, save_path


def confusion_insights(
    matrix: np.ndarray,
    class_names: list[str],
    top_k: int = 10,
) -> dict:
    diag = matrix.diagonal().astype(float)
    support = matrix.sum(axis=1).astype(float)
    class_acc = np.divide(diag, np.maximum(support, 1))
    best = np.argsort(class_acc)[::-1][:top_k]
    worst = np.argsort(class_acc)[:top_k]

    off_diag = matrix.copy().astype(float)
    np.fill_diagonal(off_diag, 0)
    pairs = []
    flat_idx = np.dstack(np.unravel_index(np.argsort(off_diag.ravel())[::-1], off_diag.shape))[0]
    for true_idx, pred_idx in flat_idx[:top_k]:
        if off_diag[true_idx, pred_idx] <= 0:
            break
        pairs.append(
            {
                "true": class_names[int(true_idx)],
                "predicted": class_names[int(pred_idx)],
                "count": int(off_diag[true_idx, pred_idx]),
                "true_glyph": glyph_only(class_names[int(true_idx)]),
                "predicted_glyph": glyph_only(class_names[int(pred_idx)]),
            }
        )
    return {
        "best_classes": [
            {"class": class_names[i], "accuracy": float(class_acc[i])} for i in best
        ],
        "worst_classes": [
            {"class": class_names[i], "accuracy": float(class_acc[i])} for i in worst
        ],
        "confused_pairs": pairs,
    }


def per_class_table(report: dict, class_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in class_names:
        item = report.get(name, {})
        rows.append(
            {
                "Character": glyph_only(name),
                "Class": name,
                "Precision": item.get("precision"),
                "Recall": item.get("recall"),
                "F1 Score": item.get("f1-score"),
                "Support": item.get("support"),
            }
        )
    return pd.DataFrame(rows)
