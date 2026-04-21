"""
src/ensemble/weighted_avg.py
─────────────────────────────
Weighted average ensemble — comparison baseline for the stacking ensemble.

Each base model's calibrated probability is multiplied by a weight, and the
weighted sum is the ensemble's output.

Weights are optimised on the validation set by minimising log-loss (NLL)
using scipy's L-BFGS-B. Weights are constrained to [0, 1] and normalised
to sum to 1 after optimisation.

This is a much simpler ensemble than stacking. It often achieves competitive
AUPRC, and if it matches the stacking ensemble it suggests the base models'
calibrated probabilities are already well-scaled and the meta-learner isn't
adding much. In practice, stacking usually wins by a small margin because the
meta-learner can learn that some models are more reliable in specific score
ranges.

We keep both and pick whichever has better AUPRC + lower ECE on the val set.
The winner becomes the production ensemble in stages 5–8.
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from src.calibration.calibrators import (
    IsotonicCalibrator,
    cross_val_calibration,
)
from src.models import ENSEMBLE_MODEL_NAMES


class WeightedAverageEnsemble:
    """
    Weighted average of calibrated base model probabilities.

    Weights are optimised on the val set to minimise NLL.
    The final output is calibrated using cross-validated isotonic regression.
    """

    name = "weighted_avg_ensemble"

    def __init__(self, calibrated_models: dict, random_state: int = 42):
        self.calibrated_models = calibrated_models
        self.model_names       = [n for n in ENSEMBLE_MODEL_NAMES if n in calibrated_models]
        self.random_state      = random_state
        self._executor         = None   # initialised lazily and reused

        self.weights_         = None
        self.meta_calibrator_ = None

    def _get_executor(self):
        """Returns a persistent thread pool — created once, reused across requests.
        Uses object.__getattribute__ defensively to handle objects loaded from
        joblib that were pickled before _executor was added to __init__."""
        try:
            executor = object.__getattribute__(self, '_executor')
        except AttributeError:
            executor = None
        if executor is None:
            from concurrent.futures import ThreadPoolExecutor
            executor = ThreadPoolExecutor(max_workers=len(self.model_names))
            object.__setattr__(self, '_executor', executor)
        return executor

    def fit(
        self,
        val_matrix: np.ndarray,
        y_val: np.ndarray,
    ) -> "WeightedAverageEnsemble":
        """
        Optimises weights on the validation set.

        Args:
            val_matrix: (n_val × n_models) calibrated val predictions.
            y_val:      True val labels.
        """
        n_models = val_matrix.shape[1]

        def neg_log_likelihood(w: np.ndarray) -> float:
            w = np.clip(w, 1e-6, None)
            w = w / w.sum()
            probs = np.clip((val_matrix * w).sum(axis=1), 1e-7, 1 - 1e-7)
            return -np.mean(y_val * np.log(probs) + (1 - y_val) * np.log(1 - probs))

        # Initialise with equal weights
        w0 = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0)] * n_models
        result = minimize(
            neg_log_likelihood, w0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        raw_w = np.clip(result.x, 0.0, None)
        self.weights_ = raw_w / raw_w.sum()

        print(f"\n    Optimised weights:")
        for name, w in zip(self.model_names, self.weights_):
            print(f"      {name:<25}  weight = {w:.4f}")

        # Calibrate the ensemble output on val
        raw_ensemble_val = (val_matrix * self.weights_).sum(axis=1)
        print(f"\n    Calibrating weighted average output (cross-val isotonic)...")
        self.meta_calibrator_ = cross_val_calibration(
            raw_ensemble_val, y_val,
            calibrator_cls=IsotonicCalibrator,
            n_splits=5,
        )
        return self

    def build_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        cols = [
            self.calibrated_models[name].predict_proba_calibrated(X)
            for name in self.model_names
        ]
        return np.column_stack(cols).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feat_matrix  = self.build_feature_matrix(X)
        raw_ensemble = (feat_matrix * self.weights_).sum(axis=1)
        return self.meta_calibrator_.predict(raw_ensemble).astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  [{self.name}] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "WeightedAverageEnsemble":
        return joblib.load(path)