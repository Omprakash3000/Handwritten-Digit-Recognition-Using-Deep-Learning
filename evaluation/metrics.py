"""Multiclass classification metrics including One-vs-Rest ROC-AUC."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["macro_roc_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
        metrics["weighted_roc_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
        )
    except ValueError:
        metrics["macro_roc_auc"] = None
        metrics["weighted_roc_auc"] = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_accuracy = []
    for idx, name in enumerate(class_names):
        mask = y_true == idx
        acc = float((y_pred[mask] == y_true[mask]).mean()) if mask.any() else 0.0
        per_class_accuracy.append({"class": name, "accuracy": acc, "support": int(mask.sum())})
        if name in report:
            report[name]["per_class_accuracy"] = acc
    metrics["classification_report"] = report
    metrics["per_class_accuracy"] = per_class_accuracy
    return metrics
