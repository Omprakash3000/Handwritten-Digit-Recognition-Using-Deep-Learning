"""About the Devanagari Handwritten Recognition System."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import PLOT_DIR, PROJECT_NAME, PROJECT_SUBTITLE


def render() -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #0f172a;">ℹ️ About {PROJECT_NAME}</h2>
            <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.95rem;">{PROJECT_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.4rem; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin: 0 0 0.8rem 0; color: #1e293b; font-size: 1.05rem;">📐 System Specifications</h4>
                <ul style="margin: 0; padding-left: 1.2rem; color: #475569; font-size: 0.92rem; line-height: 1.7;">
                    <li><strong>Multimodal Engine:</strong> Google Gemini 2.5 Vision API</li>
                    <li><strong>Deep Neural Backbones:</strong> ResNet-50, DenseNet-121, ViT-B/16</li>
                    <li><strong>Decision Mechanism:</strong> Weighted Soft Voting Ensemble</li>
                    <li><strong>Input Resolution:</strong> 32×32 Grayscale Canonical Stroke Grid</li>
                    <li><strong>Frameworks:</strong> PyTorch 2.5, Torchvision, TIMM, Transformers</li>
                    <li><strong>Frontend Platform:</strong> Streamlit with Custom CSS & Plotly</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 1.4rem; box-shadow: 0 4px 12px rgba(0,0,0,0.02);">
                <h4 style="margin: 0 0 0.8rem 0; color: #1e293b; font-size: 1.05rem;">🔒 Rigorous Research Protocol</h4>
                <ul style="margin: 0; padding-left: 1.2rem; color: #475569; font-size: 0.92rem; line-height: 1.7;">
                    <li><strong>Independent Test Split:</strong> 13,800 test images strictly isolated</li>
                    <li><strong>Validation Split:</strong> 85% train / 15% validation stratified split</li>
                    <li><strong>Deterministic Seeding:</strong> Seed 42 across PyTorch, NumPy, and CUDA</li>
                    <li><strong>Anti-Hallucination:</strong> Ground-truth metrics loaded directly from JSON</li>
                    <li><strong>Explainability:</strong> Grad-CAM layer activations & ViT self-attention</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 🏗️ End-to-End System Pipeline")
    arch = PLOT_DIR / "figure1_system_architecture.png"
    if arch.exists():
        st.image(str(arch), caption="Figure 1: Multimodal & Deep Learning Ensemble Pipeline", use_container_width=True)
    else:
        st.code(
            """
                 Handwritten Image / Scan
                            │
                            ▼
              Adaptive Stroke Preprocessing
             (Bounding Box, Inversion, 32x32)
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
          ResNet-50    DenseNet-121      ViT
             │              │              │
             └──────────────┼──────────────┘
                            ▼
                  Soft Voting Ensemble
                            │
                            ▼
                 Predicted Hindi Glyph
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
       Audio Speech Synthesis       Explainability (Grad-CAM)
            """,
            language="text",
        )
