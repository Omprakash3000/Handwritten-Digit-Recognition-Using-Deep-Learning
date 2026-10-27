"""Render the system architecture figure used in the report and UI."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from config import PLOT_DIR, ensure_output_dirs


def plot_architecture(save_path: Path | None = None) -> Path:
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / "figure1_system_architecture.png")
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_title("Figure 1. System Architecture — Handwritten Detection", fontsize=14, pad=12)

    boxes = [
        (3.0, 12.4, "Handwritten Image"),
        (3.0, 10.8, "Preprocessing"),
        (0.4, 8.8, "ResNet50"),
        (3.6, 8.8, "DenseNet121"),
        (6.8, 8.8, "Vision Transformer"),
        (3.0, 6.8, "Weighted Soft Voting Ensemble"),
        (3.0, 5.2, "Character Prediction"),
        (3.0, 3.6, "Confidence Score"),
        (3.0, 2.0, "Explainable AI Output"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.2,
                0.9,
                boxstyle="round,pad=0.03,rounding_size=0.15",
                facecolor="#eef4ff",
                edgecolor="#1f4e79",
                linewidth=1.5,
            )
        )
        ax.text(x + 1.6, y + 0.45, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((4.6, 12.4), (4.6, 11.7)),
        ((4.6, 10.8), (4.6, 10.1)),
        ((4.6, 10.1), (2.0, 9.7)),
        ((4.6, 10.1), (5.2, 9.7)),
        ((4.6, 10.1), (8.4, 9.7)),
        ((2.0, 8.8), (4.6, 7.7)),
        ((5.2, 8.8), (4.6, 7.7)),
        ((8.4, 8.8), (4.6, 7.7)),
        ((4.6, 6.8), (4.6, 6.1)),
        ((4.6, 5.2), (4.6, 4.5)),
        ((4.6, 3.6), (4.6, 2.9)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "color": "#1f4e79"})

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path
