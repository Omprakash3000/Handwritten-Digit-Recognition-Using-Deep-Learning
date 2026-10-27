"""One-vs-Rest ROC and precision-recall curves for 46-class classification."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from config import PLOT_DIR, ensure_output_dirs
from dataset.class_labels import glyph_only


def plot_ovr_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path | None = None,
    max_curves: int = 12,
) -> Path:
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / f"{title.replace(' ', '_').lower()}_roc.png")
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx in range(min(len(class_names), max_curves)):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], y_prob[:, idx])
        ax.plot(fpr, tpr, label=f"{glyph_only(class_names[idx])} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def plot_precision_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    title: str,
    save_path: Path | None = None,
    max_curves: int = 12,
) -> Path:
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / f"{title.replace(' ', '_').lower()}_pr.png")
    y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    for idx in range(min(len(class_names), max_curves)):
        precision, recall, _ = precision_recall_curve(y_bin[:, idx], y_prob[:, idx])
        ax.plot(recall, precision, label=glyph_only(class_names[idx]))
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path
