"""
src/calibration/calibrators.py
───────────────────────────────
Three calibration methods, each wrapping a base model.

After calibration, model.predict_proba_calibrated(X) returns a probability
that genuinely reflects the empirical fraud rate — i.e. if the model outputs
0.8 for a batch of transactions, roughly 80% of them should actually be fraud.

Why calibration matters here:
    The class imbalance (0.17% fraud) causes most models to output very low
    raw probabilities — even for genuine fraud. A Random Forest might output
    p=0.03 for a fraud case that it correctly ranks above most legitimate
    transactions. After calibration, that 0.03 becomes something like 0.45,
    which is a meaningful probability. The threshold optimisation in Stage 5
    depends on these probabilities being interpretable.

Three methods:
─────────────────────────────────────────────────────────────────────────────
1. Platt Scaling
   Fits a logistic regression on top of the model's raw scores.
   Formula: p_cal = σ(A·s + B), where s is the raw score.
   Best for: models with sigmoid-shaped score distributions (SVM, LR).
   Not used for LR (already calibrated) but kept as a comparison.

2. Isotonic Regression
   Fits a non-parametric monotone step function mapping raw scores → probs.
   More flexible than Platt but needs more data to fit reliably.
   We have ~40k val samples so this is fine.
   Best for: Random Forest, XGBoost, LightGBM, Autoencoder.

3. Temperature Scaling
   Divides the MLP's raw logit by a scalar T before the sigmoid:
       p_cal = σ(logit / T)
   T > 1 → model was overconfident, softens probabilities toward 0.5
   T < 1 → model was underconfident, sharpens probabilities toward 0/1
   T is found by minimising NLL on the validation set.
   Best for: neural networks. Preserves ranking exactly (monotone transform).
   Requires access to raw logits — only the MLP exposes predict_logits().

Cross-validated calibration:
    Because fraud is rare, fitting calibration on a single val set can be
    noisy (few positive examples per bin). We use cross_val_calibration()
    which fits the calibrator on multiple folds and averages, giving more
    stable probability estimates.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


# ─── Platt Scaling ────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Fits a logistic regression on the model's raw scores.
    p_cal = σ(A·score + B)
    """
    name = "platt"

    def __init__(self):
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._lr.fit(scores.reshape(-1, 1), y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1].astype(np.float32)


# ─── Isotonic Regression ─────────────────────────────────────────────────────

class IsotonicCalibrator:
    """
    Non-parametric monotone mapping from raw scores to calibrated probabilities.
    More flexible than Platt; recommended for tree-based models.
    """
    name = "isotonic"

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._iso.fit(scores, y)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self._iso.predict(scores).astype(np.float32)


# ─── Temperature Scaling ─────────────────────────────────────────────────────

class TemperatureCalibrator:
    """
    Divides the MLP's raw logit by a learned scalar T, then applies sigmoid.
    p_cal = σ(logit / T)

    T is found by minimising negative log-likelihood on the validation set
    via a scalar optimisation (Brent's method — exact, no gradient needed).

    Only applicable to models that expose predict_logits().
    """
    name = "temperature"

    def __init__(self):
        self.T_: float = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        """
        Args:
            logits: Raw pre-sigmoid logits from the MLP.
            y:      True binary labels.
        """
        y = y.astype(np.float64)

        def nll(T: float) -> float:
            T = max(T, 1e-3)   # numerical stability
            probs = expit(logits / T)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.T_ = float(result.x)
        print(f"      [temperature] T = {self.T_:.4f}  "
              f"({'over-confident → softened' if self.T_ > 1 else 'under-confident → sharpened'})")
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        return expit(logits / self.T_).astype(np.float32)


# ─── Calibrated model wrapper ─────────────────────────────────────────────────

class CalibratedModel:
    """
    Wraps a base model + a fitted calibrator.

    Exposes:
        .predict_proba(X)            → raw (uncalibrated) probability
        .predict_proba_calibrated(X) → calibrated probability
        .model_name
        .calibrator_name
    """

    def __init__(self, base_model, calibrator):
        self.base_model_  = base_model
        self.calibrator_  = calibrator
        self.model_name   = base_model.name
        self.calibrator_name = calibrator.name

    def predict_proba(self, X) -> np.ndarray:
        return self.base_model_.predict_proba(X)

    def predict_proba_calibrated(self, X) -> np.ndarray:
        """The output of this method is what the ensemble in Stage 4 uses."""
        raw = self.base_model_.predict_proba(X)
        return self.calibrator_.predict(raw)


# ─── Cross-validated calibration ─────────────────────────────────────────────

def cross_val_calibration(
    scores: np.ndarray,
    y: np.ndarray,
    calibrator_cls,
    n_splits: int = 5,
) -> "AveragedCalibrator":
    """
    Fits the calibrator on multiple stratified folds and returns a wrapper
    that averages predictions across all fold-calibrators.

    This is the recommended approach for this dataset because fraud is so rare
    that a single-fold calibration has too few positive examples per bin to be
    reliable. Averaging over 5 folds gives more stable probability estimates.

    Args:
        scores:         Raw model probabilities on the validation set.
        y:              True labels on the validation set.
        calibrator_cls: One of PlattCalibrator, IsotonicCalibrator.
        n_splits:       Number of stratified CV folds.

    Returns:
        AveragedCalibrator that averages across all fold calibrators.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    calibrators = []

    for fold, (train_idx, _) in enumerate(skf.split(scores, y)):
        cal = calibrator_cls()
        cal.fit(scores[train_idx], y[train_idx])
        calibrators.append(cal)

    return AveragedCalibrator(calibrators, calibrator_cls.name)


class AveragedCalibrator:
    """Averages predictions from multiple fold-calibrators."""

    def __init__(self, calibrators: list, name: str):
        self.calibrators_ = calibrators
        self.name = name

    def predict(self, scores: np.ndarray) -> np.ndarray:
        preds = np.stack([c.predict(scores) for c in self.calibrators_], axis=0)
        return preds.mean(axis=0).astype(np.float32)