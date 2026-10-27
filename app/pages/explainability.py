"""Explainable AI visualizations for handwritten Devanagari character recognition."""

from __future__ import annotations

import random
from pathlib import Path

import streamlit as st
from PIL import Image

from app.components.loaders import cached_models, missing_training_message, models_ready
from config import PLOT_DIR, discover_dataset_path
from explainability.gradcam_densenet import densenet_gradcam
from explainability.gradcam_resnet import resnet_gradcam
from explainability.vit_attention import vit_attention_map
from inference import preprocess_image


def _load_sample_for_xai() -> tuple[Image.Image, any]:
    """Load a sample character image and preprocess it into a tensor."""
    sample_dir = discover_dataset_path() / "Test" / "character_1_ka"
    if not sample_dir.exists():
        sample_dir = Path("archive") / "DevanagariHandwrittenCharacterDataset" / "Test" / "character_1_ka"
    files = list(sample_dir.glob("*.png"))
    sample_file = files[0] if files else None
    if sample_file:
        img = Image.open(sample_file).convert("RGB")
        tensor = preprocess_image(img, auto_invert=True, auto_crop=True)
        return img, tensor
    raise FileNotFoundError("Could not find sample dataset character.")


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #0f172a;">🔍 Explainable AI & Attention Saliency</h2>
            <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.95rem;">
                Interpret model decisions using Grad-CAM heatmaps for CNNs (ResNet-50, DenseNet-121) and self-attention rollout for Vision Transformers (ViT).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not models_ready():
        st.error(missing_training_message())
        _show_saved_figures()
        return

    # Check if an active tensor exists
    has_image = "last_tensor" in st.session_state and "last_image" in st.session_state

    if not has_image:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 1.5px dashed #cbd5e1; border-radius: 16px; padding: 1.5rem; text-align: center; margin-bottom: 1.5rem;">
                <p style="margin: 0 0 0.8rem 0; color: #475569; font-weight: 600;">
                    No handwritten image has been uploaded in this session yet.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("⚡ Load Demo Sample Character 'क' for Live Visual Saliency", type="primary"):
            try:
                img, tensor = _load_sample_for_xai()
                st.session_state["last_image"] = img
                st.session_state["last_tensor"] = tensor
                st.rerun()
            except Exception as e:
                st.warning(f"Could not load demo sample: {e}")

        _show_saved_figures()
        return

    models, _class_names = cached_models()
    tensor = st.session_state["last_tensor"]
    image = st.session_state["last_image"]

    choice = st.selectbox(
        "Select Explainability Technique & Backbone",
        ["Grad-CAM — ResNet-50 (Final Conv Layer)", "Grad-CAM — DenseNet-121 (DenseBlock 4)", "ViT Self-Attention Map (CLS Token Rollout)"],
    )

    try:
        with st.spinner("Computing gradient-weighted class activation mapping..."):
            if choice.startswith("Grad-CAM — ResNet-50"):
                output = resnet_gradcam(models["resnet50"], tensor)
            elif choice.startswith("Grad-CAM — DenseNet-121"):
                output = densenet_gradcam(models["densenet121"], tensor)
            else:
                output = vit_attention_map(models["vit"], tensor)
    except Exception as exc:
        st.error(f"Could not generate the explanation: {exc}")
        _show_saved_figures()
        return

    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="1. Input Character Image", use_container_width=True)
    c2.image(output["heatmap"], caption="2. Saliency Heatmap", use_container_width=True, clamp=True)
    c3.image(output["overlay"], caption="3. Visual Overlay", use_container_width=True)

    st.markdown(
        """
        <div style="background: #eef2ff; border-radius: 12px; padding: 0.9rem 1.2rem; color: #3730a3; font-size: 0.9rem; font-weight: 500; margin-top: 1rem; border: 1px solid #c7d2fe;">
            💡 <strong>Interpretation:</strong> The red/yellow luminous regions indicate strokes, loops, or the shirorekha (top horizontal line) that exerted the highest weight in driving the neural network's final class prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='margin: 2rem 0; border: none; border-top: 1px solid #e2e8f0;'>", unsafe_allow_html=True)
    _show_saved_figures()


def _show_saved_figures() -> None:
    st.markdown("### 🖼️ Saved Research Explainability Gallery")
    st.caption("Pre-computed publication figures generated during the evaluate.py benchmark run.")

    c1, c2 = st.columns(2)
    with c1:
        p1 = PLOT_DIR / "figure6_gradcam_resnet.png"
        if p1.exists():
            st.image(str(p1), caption="ResNet-50 Grad-CAM Feature Attribution", use_container_width=True)
        p3 = PLOT_DIR / "figure7_vit_attention.png"
        if p3.exists():
            st.image(str(p3), caption="Vision Transformer Patch Attention Maps", use_container_width=True)

    with c2:
        p2 = PLOT_DIR / "figure6_gradcam_densenet.png"
        if p2.exists():
            st.image(str(p2), caption="DenseNet-121 Feature Attribution", use_container_width=True)
        p4 = PLOT_DIR / "figure8_tsne_features.png"
        if p4.exists():
            st.image(str(p4), caption="t-SNE Feature Latent Space Clustering", use_container_width=True)
