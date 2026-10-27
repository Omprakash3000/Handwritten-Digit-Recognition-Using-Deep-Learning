"""Evaluate trained models and the ensemble on the independent test split."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 4))
warnings.filterwarnings("ignore", message=r".*Could not find the number of physical cores.*")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

plt.rcParams["font.sans-serif"] = ["Nirmala UI", "Mangal", "DejaVu Sans", "Arial"]
warnings.filterwarnings("ignore", message=r".*missing from font.*")

from config import (
    CLASS_NAMES_PATH,
    DENSENET_CHECKPOINT,
    DEVICE,
    ENSEMBLE_WEIGHTS_PATH,
    EVALUATION_PATH,
    PLOT_DIR,
    RESNET_CHECKPOINT,
    RESULTS_DIR,
    VIT_CHECKPOINT,
    ensure_output_dirs,
)
from dataset.class_labels import glyph_only
from dataset.dataset_split import build_splits
from ensemble.ensemble_evaluation import ablation_study, evaluate_probability_set
from ensemble.ensemble_predict import combine_probabilities
from evaluation.classification_report import save_classification_outputs
from evaluation.confusion_matrix import confusion_insights, plot_confusion_matrix
from evaluation.misclassified import collect_misclassified, plot_misclassified
from evaluation.roc_analysis import plot_ovr_roc, plot_precision_recall
try:
    from explainability.feature_visualization import extract_embeddings, plot_tsne
except Exception:
    extract_embeddings, plot_tsne = None, None
from explainability.gradcam_densenet import densenet_gradcam
from explainability.gradcam_resnet import resnet_gradcam
from explainability.vit_attention import vit_attention_map
from models.model_factory import create_model, load_checkpoint
from utils import load_json, save_json, set_seed, setup_logging

LOGGER = setup_logging("evaluate")
CHECKPOINTS = {
    "resnet50": RESNET_CHECKPOINT,
    "densenet121": DENSENET_CHECKPOINT,
    "vit": VIT_CHECKPOINT,
}


@torch.no_grad()
def collect_outputs(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    probs, labels, preds, paths = [], [], [], []
    for images, targets, batch_paths in loader:
        images = images.to(DEVICE)
        logits = model(images)
        batch_probs = F.softmax(logits, dim=1).cpu().numpy()
        batch_preds = batch_probs.argmax(axis=1)
        probs.append(batch_probs)
        labels.append(targets.numpy())
        preds.append(batch_preds)
        paths.extend(batch_paths)
    return (
        np.concatenate(probs),
        np.concatenate(preds),
        np.concatenate(labels),
        paths,
    )


def load_available_models(num_classes: int) -> dict:
    models = {}
    for name, path in CHECKPOINTS.items():
        if not path.exists():
            LOGGER.warning("Skipping %s — checkpoint missing at %s", name, path)
            continue
        model = create_model(name, num_classes=num_classes, pretrained=False)
        load_checkpoint(model, path)
        models[name] = model
    if not models:
        raise FileNotFoundError(
            "No trained checkpoints found. Run `python train.py` before evaluation."
        )
    return models


def main() -> None:
    set_seed()
    ensure_output_dirs()
    splits = build_splits()
    class_names = splits["info"].class_names
    if CLASS_NAMES_PATH.exists():
        saved = load_json(CLASS_NAMES_PATH)["class_names"]
        if saved != class_names:
            LOGGER.warning("Saved class names differ from current folders; using folder order.")
    models = load_available_models(len(class_names))

    model_probs = {}
    test_labels = None
    test_paths = None
    results = {}
    display_table = []

    for name, model in models.items():
        LOGGER.info("Evaluating %s on the independent test set", name)
        probs, preds, labels, paths = collect_outputs(model, splits["test_loader"])
        model_probs[name] = probs
        test_labels = labels
        test_paths = paths
        metrics = evaluate_probability_set(probs, labels, class_names)
        save_classification_outputs(name, metrics, class_names)
        matrix, cm_path = plot_confusion_matrix(
            labels, preds, class_names, f"Figure 4. Confusion Matrix — {name}"
        )
        insights = confusion_insights(matrix, class_names)
        roc_path = plot_ovr_roc(
            labels, probs, class_names, f"Figure 5. One-vs-Rest ROC — {name}"
        )
        pr_path = plot_precision_recall(
            labels, probs, class_names, f"Precision-Recall — {name}"
        )
        errors = collect_misclassified(paths, labels, preds, probs, class_names)
        miss_path = plot_misclassified(errors, PLOT_DIR / f"{name}_misclassified.png")
        serializable = {
            key: value
            for key, value in metrics.items()
            if key != "classification_report"
        }
        serializable["classification_report"] = metrics["classification_report"]
        results[name] = {
            "metrics": serializable,
            "confusion_matrix_path": str(cm_path),
            "roc_path": str(roc_path),
            "pr_path": str(pr_path),
            "misclassified_plot": str(miss_path) if miss_path else None,
            "insights": insights,
            "misclassified_examples": errors,
        }
        display_table.append(
            {
                "Model": name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["macro_precision"],
                "Recall": metrics["macro_recall"],
                "F1": metrics["macro_f1"],
                "ROC-AUC": metrics["macro_roc_auc"],
            }
        )

    if len(model_probs) == 3 and ENSEMBLE_WEIGHTS_PATH.exists():
        weights = load_json(ENSEMBLE_WEIGHTS_PATH).get("weights")
        ensemble_probs = combine_probabilities(model_probs, weights)
        ensemble_preds = ensemble_probs.argmax(axis=1)
        metrics = evaluate_probability_set(ensemble_probs, test_labels, class_names)
        save_classification_outputs("ensemble", metrics, class_names)
        matrix, cm_path = plot_confusion_matrix(
            test_labels, ensemble_preds, class_names, "Figure 4. Confusion Matrix — Ensemble"
        )
        insights = confusion_insights(matrix, class_names)
        roc_path = plot_ovr_roc(
            test_labels, ensemble_probs, class_names, "Figure 5. One-vs-Rest ROC — Ensemble"
        )
        pr_path = plot_precision_recall(
            test_labels, ensemble_probs, class_names, "Precision-Recall — Ensemble"
        )
        errors = collect_misclassified(
            test_paths, test_labels, ensemble_preds, ensemble_probs, class_names
        )
        miss_path = plot_misclassified(errors, PLOT_DIR / "ensemble_misclassified.png")
        results["ensemble"] = {
            "metrics": {
                key: value
                for key, value in metrics.items()
            },
            "weights": weights,
            "confusion_matrix_path": str(cm_path),
            "roc_path": str(roc_path),
            "pr_path": str(pr_path),
            "misclassified_plot": str(miss_path) if miss_path else None,
            "insights": insights,
            "misclassified_examples": errors,
        }
        display_table.append(
            {
                "Model": "ensemble",
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["macro_precision"],
                "Recall": metrics["macro_recall"],
                "F1": metrics["macro_f1"],
                "ROC-AUC": metrics["macro_roc_auc"],
            }
        )
        ablation = ablation_study(model_probs, test_labels, class_names, weights)
        results["ablation"] = ablation
        pd.DataFrame(ablation).to_csv(RESULTS_DIR / "table5_ablation_study.csv", index=False)

    pd.DataFrame(display_table).to_csv(RESULTS_DIR / "table3_model_performance.csv", index=False)

    if extract_embeddings is not None and plot_tsne is not None:
        try:
            embed_model_name = "densenet121" if "densenet121" in models else next(iter(models))
            embeddings, embed_labels = extract_embeddings(
                models[embed_model_name],
                splits["test_loader"],
                embed_model_name,
                num_classes=len(class_names),
            )
            tsne_path = plot_tsne(embeddings, embed_labels, class_names)
            results["tsne_path"] = str(tsne_path)
        except Exception as exc:
            LOGGER.warning("t-SNE visualization skipped: %s", exc)

    try:
        sample_images, sample_labels, sample_paths = next(iter(splits["test_loader"]))
        sample = sample_images[0]
        if "resnet50" in models:
            resnet_gradcam(models["resnet50"], sample, save_path=PLOT_DIR / "figure6_gradcam_resnet.png")
        if "densenet121" in models:
            densenet_gradcam(
                models["densenet121"], sample, save_path=PLOT_DIR / "figure6_gradcam_densenet.png"
            )
        if "vit" in models:
            vit_attention_map(models["vit"], sample, save_path=PLOT_DIR / "figure7_vit_attention.png")
        results["explainability_sample"] = {
            "path": sample_paths[0],
            "true_class": class_names[int(sample_labels[0])],
            "true_glyph": glyph_only(class_names[int(sample_labels[0])]),
        }
    except Exception as exc:
        LOGGER.warning("Explainability sample skipped: %s", exc)

    save_json(results, EVALUATION_PATH)
    LOGGER.info("Evaluation complete. Results saved to %s", EVALUATION_PATH)


if __name__ == "__main__":
    main()
