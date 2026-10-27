"""Persist sklearn classification reports as JSON and CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, ensure_output_dirs
from evaluation.confusion_matrix import per_class_table
from utils import save_json


def save_classification_outputs(
    model_name: str,
    metrics: dict,
    class_names: list[str],
) -> dict[str, Path]:
    ensure_output_dirs()
    json_path = RESULTS_DIR / f"{model_name}_classification_report.json"
    csv_path = RESULTS_DIR / f"{model_name}_per_class_metrics.csv"
    save_json(metrics, json_path)
    table = per_class_table(metrics["classification_report"], class_names)
    table.to_csv(csv_path, index=False)
    return {"json": json_path, "csv": csv_path}
