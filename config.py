"""Project-wide configuration for Handwritten Detection."""

from __future__ import annotations

import os
import platform
from pathlib import Path

import torch

PROJECT_NAME = "Handwritten Detection"
PROJECT_SUBTITLE = "AI-Based Devanagari Handwritten Character Recognition"

PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# Gemini Vision API Configuration for untrained/arbitrary handwritten recognition
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_CLASSES = 46
VAL_SPLIT = 0.15
RANDOM_SEED = 42
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 1
MIN_LR = 1e-6

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
PIN_MEMORY = torch.cuda.is_available()

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
CM_DIR = OUTPUT_DIR / "confusion_matrices"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
ATTENTION_DIR = OUTPUT_DIR / "attention_maps"
REPORT_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"

RESNET_CHECKPOINT = CHECKPOINT_DIR / "best_resnet50.pth"
DENSENET_CHECKPOINT = CHECKPOINT_DIR / "best_densenet121.pth"
VIT_CHECKPOINT = CHECKPOINT_DIR / "best_vit.pth"
ENSEMBLE_WEIGHTS_PATH = CHECKPOINT_DIR / "ensemble_weights.json"
CLASS_NAMES_PATH = CHECKPOINT_DIR / "class_names.json"
HISTORY_PATH = RESULTS_DIR / "training_history.json"
EVALUATION_PATH = RESULTS_DIR / "evaluation_results.json"
DATASET_STATS_PATH = RESULTS_DIR / "dataset_stats.json"

DATASET_CANDIDATES = [
    PROJECT_ROOT / "DevanagariHandwrittenCharacterDataset",
    PROJECT_ROOT / "archive" / "DevanagariHandwrittenCharacterDataset",
    PROJECT_ROOT / "data" / "DevanagariHandwrittenCharacterDataset",
    PROJECT_ROOT / "handwritten_detection" / "DevanagariHandwrittenCharacterDataset",
]


def discover_dataset_path() -> Path:
    """Return the existing Devanagari dataset directory."""
    env_path = os.environ.get("DEVANAGARI_DATASET_PATH")
    if env_path:
        candidate = Path(env_path)
        if _is_valid_dataset(candidate):
            return candidate.resolve()

    for candidate in DATASET_CANDIDATES:
        if _is_valid_dataset(candidate):
            return candidate.resolve()

    for match in PROJECT_ROOT.rglob("DevanagariHandwrittenCharacterDataset"):
        if _is_valid_dataset(match):
            return match.resolve()

    raise FileNotFoundError(
        "Could not find DevanagariHandwrittenCharacterDataset. "
        "Place the dataset under the project folder with Train/ and Test/ splits."
    )


def _is_valid_dataset(path: Path) -> bool:
    return path.is_dir() and (path / "Train").is_dir() and (path / "Test").is_dir()


def ensure_output_dirs() -> None:
    """Create output directories used by training, evaluation, and the app."""
    for directory in [
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        PLOT_DIR,
        CM_DIR,
        GRADCAM_DIR,
        ATTENTION_DIR,
        REPORT_DIR,
        RESULTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
