"""CLI wrapper for the weighted soft-voting ensemble."""

from __future__ import annotations

import argparse
import json
import sys

from inference import predict_image


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Ensemble prediction for one image.")
    parser.add_argument("image", type=str, help="Path to a handwritten character image.")
    args = parser.parse_args()
    result = predict_image(args.image)
    payload = {
        "predicted_class": result["predicted_class"],
        "predicted_glyph": result["predicted_glyph"],
        "confidence": f"{result['confidence']:.2%}",
        "models_used": list(result["weights"].keys()),
        "resnet_prediction": (
            result["resnet_prediction"]["glyph"] if result.get("resnet_prediction") else None
        ),
        "densenet_prediction": (
            result["densenet_prediction"]["glyph"] if result.get("densenet_prediction") else None
        ),
        "vit_prediction": (
            result["vit_prediction"]["glyph"] if result.get("vit_prediction") else None
        ),
        "top5_predictions": [
            f"{p['glyph']} ({p['predicted_class']}): {p['probability']:.2%}"
            for p in result["top5_predictions"]
        ],
    }
    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    except UnicodeEncodeError:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
