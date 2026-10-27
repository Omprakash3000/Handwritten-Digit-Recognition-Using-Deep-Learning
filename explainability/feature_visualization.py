"""t-SNE / PCA visualization of deep feature embeddings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from config import DEVICE, PLOT_DIR, ensure_output_dirs
from dataset.class_labels import glyph_only
from utils import setup_logging

LOGGER = setup_logging(__name__)


def _feature_extractor(model: nn.Module, name: str) -> nn.Module:
    if name == "resnet50":
        return nn.Sequential(*list(model.children())[:-1])
    if name == "densenet121":
        return nn.Sequential(model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
    return model


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    loader,
    model_name: str,
    max_per_class: int = 30,
    num_classes: int = 46,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    extractor = _feature_extractor(model, model_name).to(DEVICE)
    counts = np.zeros(num_classes, dtype=int)
    feats = []
    labels = []
    for images, targets, _paths in loader:
        images = images.to(DEVICE)
        if model_name == "vit":
            batch_feats = model.forward_features(images)
            if batch_feats.dim() == 3:
                batch_feats = batch_feats[:, 0]
        else:
            batch_feats = extractor(images)
            batch_feats = torch.flatten(batch_feats, 1)
        for feat, label in zip(batch_feats.cpu(), targets):
            cls = int(label)
            if counts[cls] >= max_per_class:
                continue
            feats.append(feat.numpy())
            labels.append(cls)
            counts[cls] += 1
        if np.all(counts >= max_per_class):
            break
    return np.stack(feats), np.array(labels)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
) -> Path:
    ensure_output_dirs()
    save_path = save_path or (PLOT_DIR / "figure8_tsne_features.png")
    pca = PCA(n_components=min(50, embeddings.shape[1]), random_state=42)
    reduced = pca.fit_transform(embeddings)
    try:
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
        points = tsne.fit_transform(reduced)
        title = "Figure 8. t-SNE Feature Embedding Visualization"
    except Exception as exc:
        LOGGER.warning("TSNE unavailable (%s); falling back to 2D PCA for visualization.", exc)
        pca_2d = PCA(n_components=2, random_state=42)
        points = pca_2d.fit_transform(reduced)
        title = "Figure 8. PCA Feature Embedding Visualization"
    fig, ax = plt.subplots(figsize=(12, 10))
    for idx, name in enumerate(class_names):
        mask = labels == idx
        ax.scatter(points[mask, 0], points[mask, 1], s=12, label=glyph_only(name), alpha=0.75)
    ax.set_title(title)
    ax.legend(ncol=6, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved t-SNE plot to %s", save_path)
    return save_path
