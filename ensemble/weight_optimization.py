"""Grid-search ensemble weights on the validation set."""

from __future__ import annotations

import itertools

import numpy as np

from ensemble.ensemble_predict import combine_probabilities
from utils import setup_logging

LOGGER = setup_logging(__name__)
MODEL_ORDER = ("resnet50", "densenet121", "vit")


def _accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    return float((probs.argmax(axis=1) == labels).mean())


def grid_search_weights(
    val_probs: dict[str, np.ndarray],
    val_labels: np.ndarray,
    step: float = 0.1,
) -> dict[str, float]:
    """Search non-negative weights that sum to 1.0."""
    grid = np.round(np.arange(0.0, 1.0 + 1e-8, step), decimals=2)
    best_acc = -1.0
    best_weights = {name: 1.0 / len(MODEL_ORDER) for name in MODEL_ORDER}

    for w1, w2 in itertools.product(grid, grid):
        w3 = round(1.0 - w1 - w2, 2)
        if w3 < -1e-8:
            continue
        weights = {
            "resnet50": float(w1),
            "densenet121": float(w2),
            "vit": float(max(w3, 0.0)),
        }
        if abs(sum(weights.values()) - 1.0) > 0.05:
            continue
        ensemble = combine_probabilities(val_probs, weights)
        acc = _accuracy(ensemble, val_labels)
        if acc > best_acc:
            best_acc = acc
            best_weights = weights

    LOGGER.info("Best validation ensemble weights: %s (acc=%.4f)", best_weights, best_acc)
    return {"weights": best_weights, "validation_accuracy": best_acc}
