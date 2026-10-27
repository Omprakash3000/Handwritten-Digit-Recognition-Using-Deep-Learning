"""Cached loaders for models, metrics, and dataset metadata."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config import DATASET_STATS_PATH, EVALUATION_PATH, HISTORY_PATH
from inference import MissingArtifactError, load_class_names, load_ensemble_weights, load_trained_models
from utils import load_json


NOT_AVAILABLE = "Not Available — Run Training"


@st.cache_resource(show_spinner=False)
def cached_models():
    class_names = load_class_names()
    return load_trained_models(class_names), class_names


def load_optional_json(path: Path):
    if path.exists():
        return load_json(path)
    return None


def evaluation_results():
    return load_optional_json(EVALUATION_PATH)


def training_history():
    return load_optional_json(HISTORY_PATH)


def dataset_stats():
    data = load_optional_json(DATASET_STATS_PATH)
    if data:
        return data
    try:
        from dataset.dataset_analysis import compute_dataset_stats

        return compute_dataset_stats()
    except Exception as exc:
        st.error(f"Dataset could not be loaded: {exc}")
        return None


def models_ready() -> bool:
    try:
        cached_models()
        return True
    except MissingArtifactError:
        return False
    except Exception:
        return False


def missing_training_message() -> str:
    return (
        "Trained checkpoints were not found. Run `python train.py` and "
        "`python evaluate.py` to generate model weights and evaluation metrics. "
        f"{NOT_AVAILABLE}"
    )


def metric_or_na(payload: dict | None, model: str, key: str):
    if not payload or model not in payload:
        return NOT_AVAILABLE
    metrics = payload[model].get("metrics", payload[model])
    value = metrics.get(key)
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, float):
        return f"{value:.4f}"
    return value
