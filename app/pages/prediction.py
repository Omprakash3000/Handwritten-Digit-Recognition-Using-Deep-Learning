"""Character prediction interface for Devanagari Handwritten Recognition."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

from app.components.loaders import cached_models, models_ready
from config import PROJECT_NAME, REPORT_DIR, discover_dataset_path, ensure_output_dirs
from dataset.class_labels import GLYPH_BY_FOLDER
from gemini_recognition import recognize_with_gemini
from inference import (
    load_ensemble_weights,
    predict_tensor,
    prepare_character_image,
    preprocess_image,
)


def _render_plotly_chart(top5: list[dict]) -> None:
    """Render an interactive Plotly horizontal bar chart for model probabilities."""
    labels = [
        f"{item['glyph']} &nbsp;({item['predicted_class'].replace('character_', '').replace('digit_', '')})"
        for item in top5
    ][::-1]
    values = [item["probability"] * 100 for item in top5][::-1]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0.0, "#818cf8"], [0.5, "#4f46e5"], [1.0, "#06b6d4"]],
                line=dict(color="rgba(255,255,255,0.3)", width=1),
                cornerradius=8,
            ),
            text=[f"&nbsp;{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#0f172a"),
            cliponaxis=False,
            hoverinfo="x+y",
        )
    )
    fig.update_layout(
        title=dict(
            text="<b>Top-5 Neural Ensemble Probabilities</b>",
            font=dict(family="Plus Jakarta Sans, sans-serif", size=14, color="#1e293b"),
            x=0.01,
        ),
        xaxis=dict(
            title=dict(text="Confidence (%)", font=dict(family="Plus Jakarta Sans, sans-serif", size=11, color="#64748b")),
            range=[0, 108],
            gridcolor="#f1f5f9",
            showline=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(family="Noto Sans Devanagari, Plus Jakarta Sans, sans-serif", size=12, color="#334155"),
            showgrid=False,
            showline=False,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=10, r=40, t=35, b=25),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_audio_pronunciation(glyph: str) -> None:
    """Render browser-native audio pronunciation using HTML5 SpeechSynthesis."""
    safe_glyph = glyph.replace('"', '\\"').replace("'", "\\'")
    html_code = f"""
    <div style="margin: 0.5rem 0 0.8rem 0; display: flex; align-items: center; gap: 10px;">
        <button id="speak-btn" onclick="speakHindi()" style="
            background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
            color: #ffffff;
            border: none;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 700;
            border-radius: 12px;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 7px;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25);
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
            transition: all 0.2s ease;
        ">
            <span>🔊</span> Listen Audio ({safe_glyph})
        </button>
        <span id="speech-status" style="font-size: 12px; color: #64748b; font-family: 'Plus Jakarta Sans', sans-serif;"></span>
    </div>
    <script>
    function speakHindi() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance("{safe_glyph}");
            utterance.lang = 'hi-IN';
            utterance.rate = 0.85;
            const status = document.getElementById('speech-status');
            utterance.onstart = () => {{ status.innerText = "Playing pronunciation..."; }};
            utterance.onend = () => {{ status.innerText = ""; }};
            utterance.onerror = () => {{ status.innerText = "Speech audio unavailable"; }};
            window.speechSynthesis.speak(utterance);
        }} else {{
            document.getElementById('speech-status').innerText = "Speech synthesis not supported in browser";
        }}
    }}
    </script>
    """
    components.html(html_code, height=48)


def _render_recognition_card(data: dict) -> None:
    """Render the primary AI recognition card with rich styling."""
    glyph = data.get("glyph", "—")
    translit = data.get("transliteration", "N/A")
    cat = data.get("type", "Character")
    conf = data.get("confidence", "High")
    meaning = data.get("meaning_en", "N/A")
    pronounce = data.get("pronunciation", "N/A")
    strokes = data.get("strokes_and_features", "")
    examples = data.get("examples", [])

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-card-header">
                <div>
                    <span class="badge-tag badge-blue">📌 {cat}</span>
                    <span class="badge-tag badge-green">✨ Verified AI Output</span>
                </div>
                <div class="confidence-pill">⚡ Confidence: {conf}</div>
            </div>
            <div style="display: flex; gap: 1.8rem; align-items: center; margin-bottom: 1.2rem;">
                <div class="glyph-showcase">{glyph}</div>
                <div>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; line-height: 1.1;">{glyph}</div>
                    <div style="margin: 0.35rem 0; font-size: 1.15rem; color: #475569;">
                        <strong>Phonetic:</strong> <em style="color: #4f46e5; font-weight: 600;">/{translit}/</em>
                    </div>
                    <div style="font-size: 0.95rem; color: #64748b;">
                        <strong>Pronunciation Guide:</strong> {pronounce}
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Audio pronunciation button
    _render_audio_pronunciation(glyph)

    st.markdown(
        f"""
            <div style="border-top: 1px solid #f1f5f9; padding-top: 1rem; margin-top: 0.5rem;">
                <div class="info-section-title">📖 Linguistic Meaning & Definition</div>
                <div class="info-section-desc">{meaning}</div>
                {f'<div class="info-section-title">✍️ Stroke & Handwriting Observation</div><div class="info-section-desc">{strokes}</div>' if strokes else ''}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if examples:
        st.markdown(
            "<div style='font-size: 1rem; font-weight: 700; color: #1e293b; margin-top: 1.2rem; margin-bottom: 0.6rem;'>📚 Hindi Words & Vocabulary Usage</div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(min(len(examples), 3))
        for idx, ex in enumerate(examples):
            with cols[idx % len(cols)]:
                st.markdown(f"<div class='vocab-chip'><strong>{ex}</strong></div>", unsafe_allow_html=True)


def _generate_report(ai_result: dict, local_result: dict | None = None) -> str:
    """Generate a clean text report for export."""
    lines = [
        f"Project: {PROJECT_NAME}",
        f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
        "=" * 60,
        "HANDWRITTEN DEVANAGARI RECOGNITION REPORT",
        f"  - Recognized Character / Word: {ai_result.get('glyph')}",
        f"  - Phonetic Transliteration: {ai_result.get('transliteration')}",
        f"  - Category: {ai_result.get('type')}",
        f"  - Confidence: {ai_result.get('confidence')}",
        f"  - Pronunciation: {ai_result.get('pronunciation')}",
        f"  - Meaning: {ai_result.get('meaning_en')}",
        f"  - Stroke Details: {ai_result.get('strokes_and_features')}",
        "=" * 60,
    ]
    if local_result:
        lines.extend([
            "DEEP LEARNING MODEL BREAKDOWN:",
            f"  - Ensemble Prediction: {local_result.get('predicted_glyph')} ({local_result.get('predicted_class')})",
            f"  - Ensemble Confidence: {local_result.get('confidence', 0):.4%}",
            f"  - ResNet50: {local_result.get('resnet_prediction', {}).get('glyph', 'N/A')}",
            f"  - DenseNet121: {local_result.get('densenet_prediction', {}).get('glyph', 'N/A')}",
            f"  - ViT: {local_result.get('vit_prediction', {}).get('glyph', 'N/A')}",
        ])
    return "\n".join(lines)


def render() -> None:
    """Main render entrypoint for the character prediction page."""
    uploaded = st.file_uploader(
        "Upload a photo or scan of any handwritten Hindi / Devanagari character or word",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload any clear handwritten photo (single character, compound conjunct, numeral, or word).",
    )

    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.session_state["active_image"] = image
            st.session_state["active_source"] = f"Uploaded File: {uploaded.name}"
        except UnidentifiedImageError:
            st.error("The uploaded file is not a valid image format. Please upload a PNG, JPG, or WEBP file.")
            return
        except Exception as exc:
            st.error(f"Error loading image: {exc}")
            return
    else:
        st.session_state.pop("active_image", None)
        st.session_state.pop("active_source", None)

    if "active_image" not in st.session_state:
        st.markdown(
            """
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 20px; padding: 3.5rem 2rem; text-align: center; color: #64748b; margin-top: 1rem;">
                <div style="font-size: 3.5rem; margin-bottom: 0.8rem;">✍️</div>
                <div style="font-weight: 700; font-size: 1.3rem; color: #1e293b;">Ready for Handwritten Input</div>
                <div style="font-size: 0.95rem; margin-top: 0.5rem; max-width: 520px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                    Drag & drop or browse a photo/scan of any handwritten Hindi character, conjunct, numeral, or word above to initiate AI recognition.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    active_img = st.session_state["active_image"]
    source_label = st.session_state.get("active_source", "Handwritten Sample")

    col_left, col_right = st.columns([1, 1.45], gap="large")

    with col_left:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #0f172a;">📷 Input Visual</h3>
                <span style="font-size: 0.8rem; font-weight: 600; color: #64748b; background: #f1f5f9; padding: 0.2rem 0.6rem; border-radius: 8px;">{source_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(active_img, caption="Original Handwritten Input", use_container_width=True)

        with st.expander("🛠️ Advanced Stroke Preprocessing", expanded=False):
            auto_invert = st.checkbox("Invert dark ink on light paper", value=True)
            auto_crop = st.checkbox("Auto-crop & center character bounding box", value=True)

        prepared = prepare_character_image(active_img, auto_invert=auto_invert, auto_crop=auto_crop)
        st.image(prepared, caption="Normalized Stroke Geometry (32x32 Canonical)", use_container_width=True)

    with col_right:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; color: #0f172a;">🎯 Recognition Analysis</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.spinner("Analyzing handwritten stroke features with Vision AI..."):
            ai_result = None
            try:
                ai_result = recognize_with_gemini(image=active_img)
            except Exception as exc:
                st.warning(f"Gemini Vision API notice: {exc}. Falling back to local deep learning ensemble.")

        # Local neural network analysis if trained
        local_result = None
        if models_ready():
            try:
                models, class_names = cached_models()
                tensor = preprocess_image(active_img, auto_invert=auto_invert, auto_crop=auto_crop)
                st.session_state["last_tensor"] = tensor
                st.session_state["last_image"] = active_img
                local_result = predict_tensor(tensor, models, class_names, load_ensemble_weights())
            except Exception as exc:
                st.warning(f"Local model inference warning: {exc}")

        # If Gemini returned data, render primary card
        if ai_result:
            _render_recognition_card(ai_result)
        elif local_result:
            # Fallback card generated from local ensemble
            fallback_data = {
                "glyph": local_result.get("predicted_glyph", "—"),
                "transliteration": local_result.get("predicted_class", "").replace("character_", "").replace("digit_", ""),
                "type": "Trained Dataset Class",
                "confidence": f"{local_result.get('confidence', 0):.1%}",
                "meaning_en": f"Devanagari character '{local_result.get('predicted_glyph', '')}' recognized by Deep Learning Ensemble.",
                "pronunciation": f"Standard Devanagari sound for {local_result.get('predicted_glyph', '')}",
                "strokes_and_features": "Identified using consensus soft-voting from ResNet50, DenseNet121, and ViT.",
                "examples": [],
            }
            _render_recognition_card(fallback_data)
        else:
            st.error("Neither Gemini Vision nor local model weights could process this image.")
            return

        # Deep Learning Tri-Model Consensus Grid
        if local_result:
            st.markdown(
                "<div style='font-size: 1rem; font-weight: 700; color: #1e293b; margin-top: 1.4rem; margin-bottom: 0.6rem;'>🧠 Deep Learning Ensemble Breakdown</div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3 = st.columns(3)
            with m1:
                res_pred = local_result.get("resnet_prediction", {})
                st.markdown(
                    f"""
                    <div class="model-metric-card">
                        <div class="model-metric-name">ResNet-50</div>
                        <div class="model-metric-glyph">{res_pred.get('glyph', '—')}</div>
                        <div class="model-metric-conf">{res_pred.get('confidence', 0):.1%} Conf</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                dense_pred = local_result.get("densenet_prediction", {})
                st.markdown(
                    f"""
                    <div class="model-metric-card">
                        <div class="model-metric-name">DenseNet-121</div>
                        <div class="model-metric-glyph">{dense_pred.get('glyph', '—')}</div>
                        <div class="model-metric-conf">{dense_pred.get('confidence', 0):.1%} Conf</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                vit_pred = local_result.get("vit_prediction", {})
                st.markdown(
                    f"""
                    <div class="model-metric-card">
                        <div class="model-metric-name">Vision Transformer</div>
                        <div class="model-metric-glyph">{vit_pred.get('glyph', '—')}</div>
                        <div class="model-metric-conf">{vit_pred.get('confidence', 0):.1%} Conf</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Interactive Plotly probability breakdown
            if local_result.get("top5_predictions"):
                st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
                _render_plotly_chart(local_result["top5_predictions"])

        # Generate & Download Report
        ensure_output_dirs()
        effective_ai = ai_result or {
            "glyph": local_result.get("predicted_glyph", "—"),
            "transliteration": local_result.get("predicted_class", ""),
            "type": "Devanagari Character",
            "confidence": f"{local_result.get('confidence', 0):.2%}",
            "pronunciation": "Standard pronunciation",
            "meaning_en": "Recognized via deep neural ensemble",
            "strokes_and_features": "Local deep learning evaluation",
        }
        report_text = _generate_report(effective_ai, local_result)
        report_path = REPORT_DIR / "latest_prediction_report.txt"
        report_path.write_text(report_text, encoding="utf-8")

        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            "📥 Export Full Recognition Report (.txt)",
            data=report_text.encode("utf-8"),
            file_name="handwritten_devanagari_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
