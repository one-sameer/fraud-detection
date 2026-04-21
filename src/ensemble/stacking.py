"""
src/ensemble/stacking.py
────────────────────────
Stacked generalisation ensemble.

How it works:
─────────────────────────────────────────────────────────────────────────────
1. Out-of-fold (OOF) prediction generation
   The training set is split into K folds (K=5).
   For each fold k:
     - Train each base model on the other K-1 folds
     - Generate calibrated probabilities on fold k (the held-out fold)
   After all folds, every training sample has a calibrated probability from
   every base model, computed on data the model never saw.
   This produces an (n_train × n_models) OOF matrix.

   Why OOF and not just val set predictions?
   If we trained the meta-learner on val-set predictions from models that
   were trained on all of the training data, the meta-learner would learn
   on in-sample predictions from the base models — it would overfit to
   their training artefacts, not their generalisation behaviour.
   OOF predictions represent genuine held-out performance, so the
   meta-learner learns the right thing.

2. Meta-learner training
   A logistic regression is trained on the (n_train × n_models) OOF matrix.
   Logistic regression is intentionally simple — the meta-learner should not
   overfit; its job is to find the optimal linear combination of the base
   models' opinions. A more complex meta-learner tends to memorise OOF noise.

3. Meta-learner calibration
   The meta-learner's output is itself calibrated using cross-validated
   isotonic regression on the validation set, exactly as in Stage 3.

4. Final inference
   At inference time:
     - Each base model (in its full, non-OOF form) generates a calibrated
       probability for the new transaction.
     - These 6 probabilities become a feature vector fed to the meta-learner.
     - The meta-learner outputs a single calibrated probability p ∈ [0, 1].
     - This p is what the decision layer in Stages 5 and 6 operates on.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.calibration.calibrators import (
    IsotonicCalibrator,
    cross_val_calibration,
    CalibratedModel,
    AveragedCalibrator,
)
from src.data.preprocessing import apply_smote
from src.models import ENSEMBLE_MODEL_NAMES


# ─── OOF generation ───────────────────────────────────────────────────────────

def generate_oof_predictions(
    base_models: dict,
    calibrated_models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generates out-of-fold calibrated predictions for the training set.

    For each fold, we retrain each base model from scratch on the in-fold
    data (with SMOTE), then apply the calibration from the full-data
    calibrated model — specifically we reuse the calibrator fitted in Stage 3.

    Reusing the Stage 3 calibrator (rather than fitting a new one per fold)
    is a pragmatic choice: refitting calibration per fold on small folds with
    very few fraud examples is noisy. The Stage 3 calibrator was fitted on
    the validation set which is completely separate, so it doesn't leak.

    Args:
        base_models:       { name: unfitted_model_class_instances } — used as
                           templates to get fresh instances per fold.
        calibrated_models: { name: CalibratedModel } — calibrators reused.
        X_train, y_train:  Full training set (preprocessed, no SMOTE yet).
        n_splits:          Number of CV folds.

    Returns:
        oof_matrix: np.ndarray of shape (n_train, n_models), float32
                    Column order matches list(base_models.keys()).
    """
    model_names = list(ENSEMBLE_MODEL_NAMES)
    n = len(X_train)
    oof_matrix = np.zeros((n, len(model_names)), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n    Fold {fold_idx + 1}/{n_splits}  "
              f"(train={len(train_idx):,}, val={len(val_idx):,}, "
              f"fraud_in_val={y_train[val_idx].sum()})")

        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        X_fold_val                 = X_train[val_idx]

        # Apply SMOTE to in-fold training data only
        X_fold_smote, y_fold_smote = apply_smote(
            X_fold_train, y_fold_train, sampling_strategy=0.1
        )

        for col_idx, name in enumerate(model_names):
            # Instantiate a fresh copy of the model with the same hyperparams
            fresh_model = _clone_model(base_models[name])

            # Train
            if name in ("xgboost", "lightgbm", "mlp"):
                fresh_model.fit(
                    X_fold_smote, y_fold_smote,
                    X_val=X_fold_val, y_val=y_train[val_idx],
                )
            elif name == "autoencoder":
                fresh_model.fit(X_fold_train, y_fold_train,
                                X_val=X_fold_val, y_val=y_train[val_idx])
            else:
                fresh_model.fit(X_fold_smote, y_fold_smote)

            # Generate raw probs on held-out fold
            if name == "mlp":
                raw = _sigmoid(fresh_model.predict_logits(X_fold_val))
            else:
                raw = fresh_model.predict_proba(X_fold_val)

            # Apply Stage 3 calibrator (reused, not refit)
            calibrator = calibrated_models[name].calibrator_
            if hasattr(calibrator, "name") and calibrator.name == "temperature":
                logits = fresh_model.predict_logits(X_fold_val)
                cal_probs = calibrator.predict(logits)
            else:
                cal_probs = calibrator.predict(raw)

            oof_matrix[val_idx, col_idx] = cal_probs

        # Progress: show per-fold average fraud probability
        fold_fraud_mean = oof_matrix[val_idx].mean(axis=0)
        print(f"    Mean cal_probs on val fold (per model): "
              + "  ".join(f"{p:.4f}" for p in fold_fraud_mean))

    return oof_matrix


def _clone_model(model):
    """Returns a new instance of the same model class with the same init params."""
    cls = type(model)
    # Collect all init params that are stored as instance attributes
    import inspect
    sig = inspect.signature(cls.__init__)
    params = {}
    for param_name in sig.parameters:
        if param_name == "self":
            continue
        if hasattr(model, param_name):
            params[param_name] = getattr(model, param_name)
    return cls(**params)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


# ─── Meta-learner ─────────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Stacking ensemble: calibrated base models + logistic regression meta-learner
    + meta-learner calibration.

    The full inference pipeline:
        raw transaction → [base model 1 calibrated prob, ..., base model 6 prob]
                        → meta-learner logistic regression
                        → meta-calibrated probability p ∈ [0, 1]

    Attributes:
        model_names:        Ordered list of base model names.
        calibrated_models:  { name: CalibratedModel } from Stage 3.
        meta_learner_:      Fitted LogisticRegression on OOF matrix.
        meta_calibrator_:   Fitted AveragedCalibrator on val set.
        meta_coefs_:        LR coefficients (interpretable weights per model).
    """

    name = "stacking_ensemble"

    def __init__(
        self,
        calibrated_models: dict,
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.calibrated_models = calibrated_models
        self.model_names       = list(calibrated_models.keys())
        self.C                 = C
        self.random_state      = random_state

        self.meta_learner_    = None
        self.meta_calibrator_ = None
        self.meta_coefs_      = None

    def fit_meta(
        self,
        oof_matrix: np.ndarray,
        y_train: np.ndarray,
        val_matrix: np.ndarray,
        y_val: np.ndarray,
    ) -> "StackingEnsemble":
        """
        Trains the meta-learner on OOF predictions and calibrates it on val.

        Args:
            oof_matrix:  (n_train × n_models) OOF calibrated predictions.
            y_train:     Training labels (same order as oof_matrix rows).
            val_matrix:  (n_val × n_models) calibrated val predictions.
            y_val:       Validation labels.
        """
        print(f"\n    Training meta-learner (LR) on OOF matrix {oof_matrix.shape}...")
        self.meta_learner_ = LogisticRegression(
            C=self.C,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=self.random_state,
        )
        self.meta_learner_.fit(oof_matrix, y_train)
        self.meta_coefs_ = dict(zip(
            self.model_names,
            self.meta_learner_.coef_[0].tolist()
        ))

        # Raw meta-learner val probs (before calibration)
        raw_meta_val = self.meta_learner_.predict_proba(val_matrix)[:, 1]

        print(f"    Meta-learner coefficients:")
        for name, coef in self.meta_coefs_.items():
            print(f"      {name:<25}  coef = {coef:+.4f}")

        # Calibrate meta-learner on val set
        print(f"\n    Calibrating meta-learner (cross-validated isotonic)...")
        self.meta_calibrator_ = cross_val_calibration(
            raw_meta_val, y_val,
            calibrator_cls=IsotonicCalibrator,
            n_splits=5,
        )

        return self

    def build_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Runs all base calibrated models on X and returns
        an (n × n_models) matrix of calibrated probabilities.
        This is the input to the meta-learner.
        """
        cols = []
        for name in self.model_names:
            cal_prob = self.calibrated_models[name].predict_proba_calibrated(X)
            cols.append(cal_prob)
        return np.column_stack(cols).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Full inference: X → feature matrix → meta-learner → calibrated p.
        This is the single number the decision layer in Stages 5/6 uses.
        """
        feat_matrix  = self.build_feature_matrix(X)
        raw_meta     = self.meta_learner_.predict_proba(feat_matrix)[:, 1]
        cal_meta     = self.meta_calibrator_.predict(raw_meta)
        return cal_meta.astype(np.float32)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  [{self.name}] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "StackingEnsemble":
        return joblib.load(path)
