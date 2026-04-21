"""
src/utils/mlflow_utils.py
─────────────────────────
MLflow helpers used across every stage of the project.

All experiments, model artifacts, plots, and metrics are tracked here.
This is set up before any model is trained so the full run history is
available for comparison, threshold sensitivity analysis, and audit.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import mlflow
import matplotlib.pyplot as plt


def setup_mlflow(
    tracking_uri: str = "experiments/mlruns",
    experiment_name: str = "fraud_detection",
) -> None:
    """
    Configures MLflow tracking URI and sets the active experiment.
    Creates the directory if it doesn't exist.
    """
    Path(tracking_uri).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"file:{tracking_uri}")
    mlflow.set_experiment(experiment_name)
    print(f"  MLflow  →  {tracking_uri}   [experiment: {experiment_name}]")


def log_figure(fig: plt.Figure, filename: str, subdir: str = "") -> None:
    """
    Saves a matplotlib figure to a temp file and logs it as an MLflow artifact.

    Args:
        fig:      Matplotlib figure to log.
        filename: Artifact filename (e.g. "reliability_xgb.png").
        subdir:   Optional artifact subfolder (e.g. "calibration_plots").
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        fig.savefig(path, bbox_inches="tight", dpi=120)
        artifact_path = subdir if subdir else None
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_numpy(arr, filename: str, subdir: str = "") -> None:
    """Saves a numpy array to disk and logs it as an artifact."""
    import numpy as np
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        np.save(path, arr)
        artifact_path = subdir if subdir else None
        mlflow.log_artifact(path + ".npy", artifact_path=artifact_path)


def log_metrics_dict(metrics: dict, prefix: str = "") -> None:
    """
    Logs a flat metrics dictionary to MLflow.
    Skips non-numeric values silently.

    Args:
        metrics: Dict of {metric_name: value}.
        prefix:  Optional prefix added to every key (e.g. "val_").
    """
    for k, v in metrics.items():
        try:
            mlflow.log_metric(f"{prefix}{k}", float(v))
        except (TypeError, ValueError):
            pass  # skip non-numeric entries (e.g. confusion matrix strings)


def end_run_if_active() -> None:
    """Ends any currently active MLflow run. Safe to call even if no run is active."""
    if mlflow.active_run() is not None:
        mlflow.end_run()
