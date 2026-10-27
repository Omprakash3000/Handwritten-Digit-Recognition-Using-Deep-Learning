"""Attention-map visualization for the timm Vision Transformer."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import ATTENTION_DIR, IMAGE_SIZE, ensure_output_dirs
from explainability.gradcam_common import denormalize, overlay_heatmap


def _attention_from_qkv(module, x: torch.Tensor) -> torch.Tensor:
    b, n, c = x.shape
    qkv = module.qkv(x).reshape(b, n, 3, module.num_heads, c // module.num_heads)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k = qkv[0], qkv[1]
    scale = module.scale if hasattr(module, "scale") else (q.shape[-1] ** -0.5)
    attn = (q @ k.transpose(-2, -1)) * scale
    return attn.softmax(dim=-1)


def _attention_rollout(attentions: list[torch.Tensor]) -> np.ndarray:
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    for attn in attentions:
        attn_mean = attn.mean(dim=1)
        residual = attn_mean + torch.eye(attn_mean.size(-1), device=attn_mean.device)
        residual = residual / residual.sum(dim=-1, keepdim=True)
    if result.dim() == 3:
        result = result.squeeze(0)
    # Attention from [CLS] token (index 0) to all patch tokens (1 to 196)
    mask = result[0, 1:]
    grid = int(mask.numel() ** 0.5)
    mask = mask.reshape(grid, grid)
    mask = mask - mask.min()
    mask = mask / (mask.max() + 1e-8)
    return mask.detach().cpu().numpy()


def vit_attention_map(
    model,
    image_tensor: torch.Tensor,
    save_path: Path | None = None,
) -> dict:
    model.eval()
    stored: list[torch.Tensor] = []
    handles = []

    def hook(module, inputs, _output):
        stored.append(_attention_from_qkv(module, inputs[0]).detach())

    for block in model.blocks:
        handles.append(block.attn.register_forward_hook(hook))

    tensor = image_tensor.to(next(model.parameters()).device)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        _ = model(tensor)
    for handle in handles:
        handle.remove()

    attn = _attention_rollout(stored) if stored else np.ones((14, 14), dtype=np.float32)
    heatmap = F.interpolate(
        torch.tensor(attn, dtype=torch.float32)[None, None],
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()
    original = denormalize(tensor.squeeze(0).cpu())
    overlay = overlay_heatmap(original, heatmap)

    ensure_output_dirs()
    save_path = save_path or (ATTENTION_DIR / "vit_attention.png")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original Character")
    axes[1].imshow(heatmap, cmap="inferno")
    axes[1].set_title("Attention Map")
    axes[2].imshow(overlay)
    axes[2].set_title("Highlighted Important Regions")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return {"heatmap": heatmap, "overlay": overlay, "original": original, "path": str(save_path)}
