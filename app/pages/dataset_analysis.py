"""Dataset analysis and class exploration page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.loaders import dataset_stats
from config import PLOT_DIR
from dataset.class_labels import display_name
from dataset.dataset_analysis import sample_images_for_class


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #0f172a;">📚 Devanagari Dataset & Class Explorer</h2>
            <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.95rem;">
                Explore the 92,000-image Devanagari Handwritten Character benchmark dataset across 46 distinct classes (36 consonants & conjuncts, 10 numerals).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stats = dataset_stats()
    if not stats:
        st.error(
            "The Devanagari dataset could not be found. Keep "
            "`DevanagariHandwrittenCharacterDataset/Train` and `Test` inside the project."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Classes", f"{stats['num_classes']} Classes")
    c2.metric("Training Set (85%)", f"{stats['training_images']:,} Imgs")
    c3.metric("Validation Set (15%)", f"{stats['validation_images']:,} Imgs")
    c4.metric("Held-Out Test Set", f"{stats['test_images']:,} Imgs")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Balanced Class Distribution")
    dist = PLOT_DIR / "figure2_dataset_distribution.png"
    if not dist.exists():
        from dataset.dataset_analysis import plot_class_distribution
        plot_class_distribution(stats, dist)
    if dist.exists():
        st.image(str(dist), caption="Devanagari Class Distribution Across Splits", use_container_width=True)

    st.markdown("### 🖼️ Interactive Class Sample Inspector")
    col_sel, col_split = st.columns([2.5, 1])
    with col_sel:
        class_name = st.selectbox(
            "Select Devanagari Character Class to Inspect",
            stats["class_names"],
            format_func=display_name,
        )
    with col_split:
        split = st.radio("Dataset Split", ["Test", "Train"], horizontal=True)

    try:
        images = sample_images_for_class(class_name, split=split, limit=8)
    except Exception as exc:
        st.error(f"Could not load class samples: {exc}")
        return

    if not images:
        st.warning("No sample images were found for this class.")
        return

    st.caption(f"Displaying 8 authentic handwritten samples for class **{display_name(class_name)}** from the {split} partition:")
    cols = st.columns(4)
    for idx, image in enumerate(images):
        cols[idx % 4].image(image, caption=f"Sample #{idx+1}", use_container_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    with st.expander("📋 View Complete Per-Class Image Count Table", expanded=False):
        table = pd.DataFrame(stats["per_class"])
        st.dataframe(table, use_container_width=True, hide_index=True)
