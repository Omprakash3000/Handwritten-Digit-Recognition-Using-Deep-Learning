"""Grad-CAM for the ResNet50 classifier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from config import GRADCAM_DIR, ensure_output_dirs
from explainability.gradcam_common import GradCAM, denormalize, overlay_heatmap
from models.resnet50_model import resnet_target_layer


def resnet_gradcam(
    model,
    image_tensor: torch.Tensor,
    class_idx: int | None = None,
    save_path: Path | None = None,
) -> dict:
    cam_engine = GradCAM(model, resnet_target_layer(model))
    heatmap = cam_engine.generate(image_tensor, class_idx=class_idx)
    cam_engine.close()
    original = denormalize(image_tensor.squeeze(0) if image_tensor.dim() == 4 else image_tensor)
    overlay = overlay_heatmap(original, heatmap)
    ensure_output_dirs()
    save_path = save_path or (GRADCAM_DIR / "resnet50_gradcam.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return {"heatmap": heatmap, "overlay": overlay, "original": original, "path": str(save_path)}
