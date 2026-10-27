"""Shared Grad-CAM implementation for CNN backbones."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import DEVICE, IMAGENET_MEAN, IMAGENET_STD


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module, _inp, output) -> None:
        self.activations = output.detach()

    def _backward_hook(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        image_tensor = image_tensor.to(DEVICE)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor.requires_grad_(True)
        logits = self.model(image_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(image_tensor.shape[-2], image_tensor.shape[-1]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(3, 1, 1)
    image = image_tensor.detach() * std + mean
    image = image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return image


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm

    colored = cm.jet(heatmap)[..., :3]
    overlay = (1 - alpha) * image + alpha * colored
    return np.clip(overlay, 0, 1)
