"""Model performance and benchmark dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.loaders import NOT_AVAILABLE, evaluation_results, metric_or_na, training_history
from config import CM_DIR, PLOT_DIR


def _show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption}: {NOT_AVAILABLE}")


def render() -> None:
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="margin: 0; font-size: 1.8rem; font-weight: 800; color: #0f172a;">📊 Deep Learning Model Benchmarks</h2>
            <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.95rem;">
                Empirical test-set evaluation across 13,800 independent Devanagari test images.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results = evaluation_results()
    history = training_history()

    # Top Metric Highlights if results exist
    if results and "ensemble" in results:
        ens_metrics = results["ensemble"].get("metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ensemble Test Accuracy", f"{ens_metrics.get('accuracy', 0):.2%}" if isinstance(ens_metrics.get('accuracy'), (int, float)) else str(ens_metrics.get('accuracy', 'N/A')))
        c2.metric("Macro F1-Score", f"{ens_metrics.get('macro_f1', 0):.2%}" if isinstance(ens_metrics.get('macro_f1'), (int, float)) else str(ens_metrics.get('macro_f1', 'N/A')))
        c3.metric("Macro Precision", f"{ens_metrics.get('macro_precision', 0):.2%}" if isinstance(ens_metrics.get('macro_precision'), (int, float)) else str(ens_metrics.get('macro_precision', 'N/A')))
        c4.metric("Macro Recall", f"{ens_metrics.get('macro_recall', 0):.2%}" if isinstance(ens_metrics.get('macro_recall'), (int, float)) else str(ens_metrics.get('macro_recall', 'N/A')))
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 🏆 Comprehensive Model Comparison")
    rows = []
    for name, label in [
        ("resnet50", "ResNet-50"),
        ("densenet121", "DenseNet-121"),
        ("vit", "Vision Transformer (ViT)"),
        ("ensemble", "Soft Voting Ensemble (Best)"),
    ]:
        rows.append(
            {
                "Architecture": label,
                "Accuracy": metric_or_na(results, name, "accuracy") if results else NOT_AVAILABLE,
                "Macro Precision": metric_or_na(results, name, "macro_precision") if results else NOT_AVAILABLE,
                "Macro Recall": metric_or_na(results, name, "macro_recall") if results else NOT_AVAILABLE,
                "Macro F1": metric_or_na(results, name, "macro_f1") if results else NOT_AVAILABLE,
                "ROC-AUC": metric_or_na(results, name, "macro_roc_auc") if results else NOT_AVAILABLE,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if results and "ablation" in results:
        st.markdown("### 🔬 Ensemble Ablation Analysis")
        st.dataframe(pd.DataFrame(results["ablation"]), use_container_width=True, hide_index=True)

    st.markdown("### 📈 Training & Validation Convergence Curves")
    if not history:
        st.info("Training history records not found yet. Run `python train.py` to generate logs.")
    else:
        model = st.selectbox("Select Model Architecture for Training Curves", list(history.keys()))
        cols = st.columns(3)
        with cols[0]:
            _show_image(PLOT_DIR / f"{model}_loss.png", "Loss Convergence (Cross-Entropy)")
        with cols[1]:
            _show_image(PLOT_DIR / f"{model}_accuracy.png", "Accuracy Trajectory (Train vs Val)")
        with cols[2]:
            _show_image(PLOT_DIR / f"{model}_learning_rate.png", "Cosine Annealing Learning Rate")

    st.markdown("### 🎯 Confusion Matrix & Class Confusion")
    selected = st.selectbox(
        "Select Model for Detailed Confusion Matrix",
        ["resnet50", "densenet121", "vit", "ensemble"],
        format_func=lambda x: {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "vit": "Vision Transformer", "ensemble": "Soft Voting Ensemble"}.get(x, x),
    )
    cm_path = None
    if results and selected in results:
        cm_path = Path(results[selected].get("confusion_matrix_path", ""))
    if cm_path and cm_path.exists():
        st.image(str(cm_path), caption=f"Normalized Confusion Matrix ({selected})", use_container_width=True)
    else:
        fallback = list(CM_DIR.glob("*.png"))
        if fallback:
            st.image(str(fallback[0]), use_container_width=True)
        else:
            st.info(NOT_AVAILABLE)

    st.markdown("### 📊 ROC & Precision-Recall Analysis")
    if results and selected in results:
        roc = Path(results[selected].get("roc_path", ""))
        pr = Path(results[selected].get("pr_path", ""))
        c1, c2 = st.columns(2)
        with c1:
            _show_image(roc, "One-vs-Rest Multiclass ROC")
        with c2:
            _show_image(pr, "Precision-Recall Curve")
    else:
        st.info(NOT_AVAILABLE)

    if results and selected in results:
        insights = results[selected].get("insights", {})
        if insights:
            st.markdown("### 💡 Error Diagnostics & Class Highlights")
            confused = insights.get("confused_pairs", [])
            if confused:
                st.markdown("##### Commonly Confused Character Pairs")
                conf_cols = st.columns(min(len(confused), 3))
                for i, pair in enumerate(confused[:3]):
                    with conf_cols[i % len(conf_cols)]:
                        st.info(f"**{pair}**")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Highest Performing Classes")
                st.write(insights.get("best_classes", []))
            with c2:
                st.markdown("##### Challenging Classes (Lowest F1)")
                st.write(insights.get("worst_classes", []))
