"""Evaluate individual models, pair ensembles, and the full ensemble."""

from __future__ import annotations

import numpy as np

from ensemble.ensemble_predict import combine_probabilities
from evaluation.metrics import compute_classification_metrics


def evaluate_probability_set(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    preds = probs.argmax(axis=1)
    return compute_classification_metrics(labels, preds, probs, class_names)


def ablation_study(
    model_probs: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: list[str],
    full_weights: dict[str, float],
) -> list[dict]:
    equal = lambda names: {n: 1.0 / len(names) for n in names}
    combinations = [
        ("ResNet50", ["resnet50"]),
        ("DenseNet121", ["densenet121"]),
        ("ViT", ["vit"]),
        ("ResNet50 + DenseNet121", ["resnet50", "densenet121"]),
        ("ResNet50 + ViT", ["resnet50", "vit"]),
        ("DenseNet121 + ViT", ["densenet121", "vit"]),
        ("ResNet50 + DenseNet121 + ViT", ["resnet50", "densenet121", "vit"]),
    ]
    rows = []
    for title, names in combinations:
        subset = {name: model_probs[name] for name in names}
        if len(names) == 3:
            weights = full_weights
        else:
            weights = equal(names)
        probs = combine_probabilities(subset, weights)
        metrics = evaluate_probability_set(probs, labels, class_names)
        rows.append(
            {
                "Architecture": title,
                "Accuracy": metrics["accuracy"],
                "Macro F1": metrics["macro_f1"],
                "ROC-AUC": metrics["macro_roc_auc"],
            }
        )
    return rows
