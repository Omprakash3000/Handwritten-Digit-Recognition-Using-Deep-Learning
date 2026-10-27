"""Soft-voting ensemble predictions."""

from __future__ import annotations

import numpy as np


def combine_probabilities(
    probability_maps: dict[str, np.ndarray],
    weights: dict[str, float],
) -> np.ndarray:
    names = list(probability_maps.keys())
    stacked = np.zeros_like(next(iter(probability_maps.values())))
    total = 0.0
    for name in names:
        weight = float(weights.get(name, 0.0))
        stacked += weight * probability_maps[name]
        total += weight
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return stacked / total


def predict_from_probs(ensemble_probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = ensemble_probs.argmax(axis=1)
    confidence = ensemble_probs.max(axis=1)
    return labels, confidence
