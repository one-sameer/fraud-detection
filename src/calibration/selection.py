"""
src/calibration/selection.py
─────────────────────────────
Selects the best calibration method for each model.

Strategy per model:
    logistic_regression → isotonic  (LR is well-calibrated; isotonic still helps)
    random_forest       → isotonic  (RF probabilities are poorly calibrated)
    xgboost             → isotonic  (same — trees produce overconfident scores)
    lightgbm            → isotonic  (same)
    mlp                 → temperature (operates on logits, cleanest for NNs)
    autoencoder         → isotonic  (reconstruction error needs remapping)

For all non-temperature models we run cross-validated calibration (5 folds)
rather than fitting on the full val set. This gives more stable estimates
because fraud cases are rare — single-fold calibration has high variance in
the positive-probability bins.

The function calibrate_all_models() returns a dict of CalibratedModel objects
that can be passed directly to Stage 4 for ensemble construction.
"""

from __future__ import annotations

import numpy as np

from src.calibration.calibrators import (
    PlattCalibrator,
    IsotonicCalibrator,
    TemperatureCalibrator,
    CalibratedModel,
    cross_val_calibration,
    AveragedCalibrator,
)
from src.evaluation.metrics import compute_ece


# Chosen calibration method per model name
CALIBRATION_STRATEGY: dict[str, str] = {
    "logistic_regression": "isotonic",
    "random_forest":       "isotonic",
    "xgboost":             "isotonic",
    "lightgbm":            "isotonic",
    "mlp":                 "isotonic",   # temperature scaling (T=1.72) hurt ECE slightly;
                                          # isotonic handles the non-uniform miscalibration better
    "autoencoder":         "isotonic",
}


def calibrate_model(
    name: str,
    base_model,
    val_scores: np.ndarray,       # raw val probs (or logits for MLP)
    val_logits: np.ndarray | None,
    y_val: np.ndarray,
) -> tuple[CalibratedModel, dict]:
    """
    Calibrates a single model and returns the wrapped CalibratedModel
    plus a dict of calibration quality metrics.

    Args:
        name:        Model name string.
        base_model:  Fitted base model with .predict_proba().
        val_scores:  Raw val probabilities from base model.
        val_logits:  Raw val logits (only needed for MLP temperature scaling).
        y_val:       True val labels.

    Returns:
        (CalibratedModel, metrics_dict)
    """
    strategy = CALIBRATION_STRATEGY[name]

    # ── ECE before calibration ──
    ece_before = compute_ece(y_val, val_scores)

    # ── fit calibrator ──
    if strategy == "temperature":
        assert val_logits is not None, "Temperature scaling requires logits."
        calibrator = TemperatureCalibrator()
        calibrator.fit(val_logits, y_val)
        cal_probs = calibrator.predict(val_logits)

    elif strategy == "isotonic":
        calibrator = cross_val_calibration(
            val_scores, y_val,
            calibrator_cls=IsotonicCalibrator,
            n_splits=5,
        )
        cal_probs = calibrator.predict(val_scores)

    elif strategy == "platt":
        calibrator = cross_val_calibration(
            val_scores, y_val,
            calibrator_cls=PlattCalibrator,
            n_splits=5,
        )
        cal_probs = calibrator.predict(val_scores)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'")

    # ── ECE after calibration ──
    ece_after = compute_ece(y_val, cal_probs)

    metrics = {
        "calibration_method": strategy,
        "ece_before":         round(ece_before, 6),
        "ece_after":          round(ece_after, 6),
        "ece_improvement":    round(ece_before - ece_after, 6),
    }

    cal_model = CalibratedModel(base_model, calibrator)
    return cal_model, metrics


def calibrate_all_models(
    base_models: dict,
    val_probs: dict[str, np.ndarray],
    val_logits: dict[str, np.ndarray | None],
    y_val: np.ndarray,
) -> tuple[dict[str, CalibratedModel], dict[str, dict]]:
    """
    Calibrates all models and returns calibrated wrappers + metrics.

    Args:
        base_models:  { name: fitted_model }
        val_probs:    { name: raw_val_probabilities }
        val_logits:   { name: val_logits_or_None }
        y_val:        True val labels.

    Returns:
        calibrated_models:  { name: CalibratedModel }
        all_cal_metrics:    { name: calibration_metrics_dict }
    """
    calibrated_models = {}
    all_cal_metrics   = {}

    for name, model in base_models.items():
        print(f"\n  ── {name} ({CALIBRATION_STRATEGY[name]}) ──")
        cal_model, metrics = calibrate_model(
            name       = name,
            base_model = model,
            val_scores = val_probs[name],
            val_logits = val_logits.get(name),
            y_val      = y_val,
        )
        print(f"      ECE  before={metrics['ece_before']:.5f}  "
              f"after={metrics['ece_after']:.5f}  "
              f"Δ={metrics['ece_improvement']:+.5f}")

        calibrated_models[name] = cal_model
        all_cal_metrics[name]   = metrics

    return calibrated_models, all_cal_metrics