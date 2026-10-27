"""Reusable inference pipeline for a single handwritten character image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from config import (
    CLASS_NAMES_PATH,
    DENSENET_CHECKPOINT,
    DEVICE,
    ENSEMBLE_WEIGHTS_PATH,
    RESNET_CHECKPOINT,
    VIT_CHECKPOINT,
)
from dataset.class_labels import glyph_only
from ensemble.ensemble_predict import combine_probabilities
from models.model_factory import create_model, load_checkpoint
from preprocessing.transforms import eval_transforms
from utils import load_json, setup_logging

LOGGER = setup_logging(__name__)
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg"}


class MissingArtifactError(FileNotFoundError):
    """Raised when a trained checkpoint or metadata file is absent."""


def load_class_names() -> list[str]:
    if not CLASS_NAMES_PATH.exists():
        from dataset.dataset_loader import load_dataset_info

        return load_dataset_info().class_names
    return load_json(CLASS_NAMES_PATH)["class_names"]


def load_ensemble_weights() -> dict[str, float]:
    if ENSEMBLE_WEIGHTS_PATH.exists():
        payload = load_json(ENSEMBLE_WEIGHTS_PATH)
        return payload.get("weights", payload)
    return {"resnet50": 1 / 3, "densenet121": 1 / 3, "vit": 1 / 3}


def prepare_character_image(
    image: Image.Image,
    auto_invert: bool = True,
    auto_crop: bool = True,
    padding_ratio: float = 0.20,
) -> Image.Image:
    """Preprocess any input image to match the dataset format (white stroke on black background, centered)."""
    gray = image.convert("L")
    arr = np.array(gray)
    h, w = arr.shape

    # 1. Background inversion detection:
    # Check borders to see if the background is light (e.g. white paper, dark pen drawings).
    if auto_invert and h > 2 and w > 2:
        border_y = max(1, h // 10)
        border_x = max(1, w // 10)
        border_pixels = np.concatenate([
            arr[0:border_y, :].flatten(),
            arr[-border_y:, :].flatten(),
            arr[:, 0:border_x].flatten(),
            arr[:, -border_x:].flatten(),
        ])
        if float(border_pixels.mean()) > 128:
            gray = ImageOps.invert(gray)
            arr = np.array(gray)

    # 2. Bounding box cropping & centered square padding:
    if auto_crop:
        threshold = max(25, int(np.percentile(arr, 75) * 0.35))
        stroke_mask = arr > threshold
        if np.any(stroke_mask):
            y_indices, x_indices = np.where(stroke_mask)
            ymin, ymax = int(y_indices.min()), int(y_indices.max())
            xmin, xmax = int(x_indices.min()), int(x_indices.max())

            cropped = gray.crop((xmin, ymin, xmax + 1, ymax + 1))
            cw, ch = cropped.size
            max_dim = max(cw, ch)
            padding = max(4, int(max_dim * padding_ratio))
            target_size = max_dim + 2 * padding

            square_img = Image.new("L", (target_size, target_size), color=0)
            paste_x = (target_size - cw) // 2
            paste_y = (target_size - ch) // 2
            square_img.paste(cropped, (paste_x, paste_y))
            gray = square_img

    return gray.convert("RGB")


def preprocess_image(
    image: Image.Image,
    auto_invert: bool = True,
    auto_crop: bool = True,
) -> torch.Tensor:
    prepared = prepare_character_image(image, auto_invert=auto_invert, auto_crop=auto_crop)
    tensor = eval_transforms()(prepared)
    return tensor.unsqueeze(0)


def load_image(source: str | Path | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    path = Path(source)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {path.suffix}. Use PNG, JPG, or JPEG.")
    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"The file is not a valid image: {path}") from exc


def load_trained_models(
    class_names: list[str] | None = None, allow_partial: bool = True
) -> dict[str, torch.nn.Module]:
    class_names = class_names or load_class_names()
    num_classes = len(class_names)
    mapping = {
        "resnet50": RESNET_CHECKPOINT,
        "densenet121": DENSENET_CHECKPOINT,
        "vit": VIT_CHECKPOINT,
    }
    models = {}
    missing = [name for name, path in mapping.items() if not path.exists()]
    if not allow_partial and missing:
        raise MissingArtifactError(
            "Missing checkpoints: "
            + ", ".join(missing)
            + ". Run `python train.py` before inference."
        )
    for name, path in mapping.items():
        if path.exists():
            model = create_model(name, num_classes=num_classes, pretrained=False)
            load_checkpoint(model, path)
            models[name] = model
    if not models:
        raise MissingArtifactError(
            f"No trained checkpoints found in {RESNET_CHECKPOINT.parent}. Run `python train.py` before inference."
        )
    return models


@torch.no_grad()
def predict_tensor(
    image_tensor: torch.Tensor,
    models: dict[str, torch.nn.Module],
    class_names: list[str],
    weights: dict[str, float] | None = None,
) -> dict:
    weights = weights or load_ensemble_weights()
    active_weights = {k: float(weights.get(k, 1.0)) for k in models.keys()}
    total_w = sum(active_weights.values())
    if total_w > 0:
        active_weights = {k: v / total_w for k, v in active_weights.items()}
    else:
        active_weights = {k: 1.0 / len(models) for k in models.keys()}

    image_tensor = image_tensor.to(DEVICE)
    probability_maps = {}
    individual = {}
    for name, model in models.items():
        model.eval()
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        probability_maps[name] = probs[None, :]
        pred_idx = int(probs.argmax())
        individual[name] = {
            "class_index": pred_idx,
            "predicted_class": class_names[pred_idx],
            "glyph": glyph_only(class_names[pred_idx]),
            "confidence": float(probs[pred_idx]),
            "probabilities": probs,
        }
    ensemble_probs = combine_probabilities(probability_maps, active_weights)[0]
    ens_idx = int(ensemble_probs.argmax())
    top5_idx = np.argsort(ensemble_probs)[::-1][:5]
    top5 = [
        {
            "class_index": int(idx),
            "predicted_class": class_names[int(idx)],
            "glyph": glyph_only(class_names[int(idx)]),
            "probability": float(ensemble_probs[int(idx)]),
        }
        for idx in top5_idx
    ]
    return {
        "predicted_class": class_names[ens_idx],
        "predicted_glyph": glyph_only(class_names[ens_idx]),
        "confidence": float(ensemble_probs[ens_idx]),
        "resnet_prediction": individual.get("resnet50"),
        "densenet_prediction": individual.get("densenet121"),
        "vit_prediction": individual.get("vit"),
        "ensemble_probabilities": ensemble_probs,
        "top5_predictions": top5,
        "class_names": class_names,
        "weights": active_weights,
    }


def predict_image(
    source: str | Path | Image.Image,
    models=None,
    use_gemini: bool = False,
    api_key: str | None = None,
) -> dict:
    image = load_image(source)
    if use_gemini:
        from gemini_recognition import recognize_with_gemini

        gemini_res = recognize_with_gemini(image, api_key=api_key)
        gemini_res["input_image"] = image
        gemini_res["prepared_image"] = prepare_character_image(image)
        return gemini_res

    class_names = load_class_names()
    models = models or load_trained_models(class_names)
    tensor = preprocess_image(image)
    result = predict_tensor(tensor, models, class_names)
    result["input_image"] = image
    result["prepared_image"] = prepare_character_image(image)
    result["input_tensor"] = tensor
    return result


if __name__ == "__main__":
    import argparse
    import json
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Predict a Devanagari handwritten character.")
    parser.add_argument("image", type=str, help="Path to handwritten image")
    parser.add_argument("--gemini", action="store_true", help="Use Gemini Vision AI for untrained characters/words")
    args = parser.parse_args()

    if args.gemini:
        output = predict_image(args.image, use_gemini=True)
        print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        sys.exit(0)

    output = predict_image(args.image)
    summary = {
        "predicted_class": output["predicted_class"],
        "predicted_glyph": output["predicted_glyph"],
        "confidence": f"{output['confidence']:.2%}",
        "models_used": list(output["weights"].keys()),
        "top5_predictions": [
            f"{p['glyph']} ({p['predicted_class']}): {p['probability']:.2%}"
            for p in output["top5_predictions"]
        ],
    }
    if output.get("resnet_prediction"):
        summary["resnet50_prediction"] = (
            f"{output['resnet_prediction']['glyph']} ({output['resnet_prediction']['predicted_class']}) - {output['resnet_prediction']['confidence']:.2%}"
        )
    if output.get("densenet_prediction"):
        summary["densenet121_prediction"] = (
            f"{output['densenet_prediction']['glyph']} ({output['densenet_prediction']['predicted_class']}) - {output['densenet_prediction']['confidence']:.2%}"
        )
    if output.get("vit_prediction"):
        summary["vit_prediction"] = (
            f"{output['vit_prediction']['glyph']} ({output['vit_prediction']['predicted_class']}) - {output['vit_prediction']['confidence']:.2%}"
        )
    try:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    except UnicodeEncodeError:
        print(json.dumps(summary, indent=2, ensure_ascii=True))
