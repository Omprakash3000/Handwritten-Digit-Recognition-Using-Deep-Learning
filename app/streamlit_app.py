"""Streamlit entry point for Handwritten Character Recognition Studio."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.components.style import APP_CSS
from app.pages import about, dataset_analysis, explainability, performance, prediction


def main() -> None:
    st.set_page_config(
        page_title="Devanagari AI Studio • Handwritten Recognition",
        page_icon="✍️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(APP_CSS, unsafe_allow_html=True)

    # Top Hero Header
    st.markdown(
        """
        <div class="hero-wrapper">
            <div class="hero-tag">
                <span>✨</span> NEXT-GEN MULTIMODAL & DEEP LEARNING ENSEMBLE
            </div>
            <h1 class="hero-title">Handwritten Character Recognition</h1>
            <p class="hero-subtitle">
                High-precision handwritten character, conjunct, and script identification.
                Powered by Google Gemini Vision multimodal intelligence paired with a tri-model deep learning ensemble
                (ResNet-50, DenseNet-121, Vision Transformer).
            </p>
            <div class="hero-pills">
                <span class="hero-pill-item">⚡ 46 Script Classes & Numerals</span>
                <span class="hero-pill-item">🧠 Soft-Voting Tri-Ensemble</span>
                <span class="hero-pill-item">🔊 Web Audio Pronunciation</span>
                <span class="hero-pill-item">🔍 Grad-CAM & ViT Attention XAI</span>
                <span class="hero-pill-item">📊 Zero-Data Fabrication Guarantee</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Clean Top Tab Navigation
    tab_pred, tab_perf, tab_xai, tab_data, tab_arch = st.tabs([
        "✍️ Recognition Studio",
        "📊 Model Benchmarks & Metrics",
        "🔍 Explainable AI & Attention",
        "📚 Dataset & Class Explorer",
        "ℹ️ System Architecture & Docs",
    ])

    with tab_pred:
        prediction.render()

    with tab_perf:
        performance.render()

    with tab_xai:
        explainability.render()

    with tab_data:
        dataset_analysis.render()

    with tab_arch:
        about.render()


if __name__ == "__main__":
    main()
