"""Home page for Handwritten Detection."""

from __future__ import annotations

import streamlit as st

from app.components.loaders import NOT_AVAILABLE, dataset_stats, evaluation_results, metric_or_na
from config import PLOT_DIR, PROJECT_NAME


def render() -> None:
    st.markdown("# Handwritten Character Detection")
    st.write(
        "This application uses deep learning models to recognize handwritten "
        "Devanagari characters from input images."
    )

    stats = dataset_stats()
    col1, col2, col3, col4, col5 = st.columns(5)
    classes = stats["num_classes"] if stats else 46
    train_images = stats["original_train_images"] if stats else 78200
    test_images = stats["test_images"] if stats else 13800
    cards = [
        (col1, classes, "Character Classes"),
        (col2, train_images, "Training Images"),
        (col3, test_images, "Test Images"),
        (col4, 3, "Deep Learning Models"),
        (col5, 1, "Ensemble Model"),
    ]
    for column, value, label in cards:
        with column:
            st.markdown(
                f"<div class='stat-card'><div class='stat-value'>{value}</div>"
                f"<div class='stat-label'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    st.subheader("Model architecture overview")
    st.markdown(
        """
        - **ResNet50** — ImageNet transfer learning with a dropout classification head.
        - **DenseNet121** — Dense connectivity backbone with a custom 46-class head.
        - **Vision Transformer (ViT-B/16)** — Patch embeddings, self-attention, and fine-tuning.
        - **Ensemble** — Weighted soft voting with validation-set weight search.
        """
    )

    arch = PLOT_DIR / "figure1_system_architecture.png"
    if not arch.exists():
        from evaluation.architecture_diagram import plot_architecture

        plot_architecture(arch)
    if arch.exists():
        st.image(str(arch), caption="System architecture")
    else:
        st.code(
            """
                 Handwritten Image
                         |
                         ↓
                  Preprocessing
                         |
                         ↓
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           ResNet50   DenseNet121   ViT
              ↓          ↓          ↓
              └──────────┼──────────┘
                         ↓
                 Soft Voting Ensemble
                         ↓
                Character Prediction
            """
        )

    st.subheader("Dataset overview")
    if stats:
        st.write(
            f"Detected dataset at `{stats['dataset_root']}` with "
            f"{stats['num_classes']} classes, {stats['training_images']} training images "
            f"after the 85/15 stratified split, {stats['validation_images']} validation images, "
            f"and {stats['test_images']} held-out test images."
        )
    else:
        st.warning("Dataset statistics are unavailable until the dataset can be discovered.")

    st.subheader("Best model result after training")
    results = evaluation_results()
    if not results:
        st.info(NOT_AVAILABLE)
        return

    best_name = None
    best_acc = -1.0
    for name in ["resnet50", "densenet121", "vit", "ensemble"]:
        if name not in results:
            continue
        acc = results[name].get("metrics", {}).get("accuracy")
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_name = name
    if best_name is None:
        st.info(NOT_AVAILABLE)
        return
    st.success(
        f"Best evaluated system: **{best_name}** with test accuracy "
        f"**{metric_or_na(results, best_name, 'accuracy')}** "
        f"and macro F1 **{metric_or_na(results, best_name, 'macro_f1')}**."
    )
    st.caption(f"{PROJECT_NAME} never displays fabricated metrics. Values come from evaluate.py.")
