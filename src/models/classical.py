"""
src/models/classical.py
───────────────────────
Classical ML models: Logistic Regression, Random Forest, XGBoost, LightGBM.

Every model is wrapped in a thin class that exposes a consistent interface:
    .fit(X, y)
    .predict_proba(X) → np.ndarray of shape (n,)  ← fraud probability only
    .save(path)  /  .load(path)

This uniform interface is what the calibration and ensemble stages depend on.
All models output a probability in [0, 1] — never a hard label.

Class imbalance is handled via scale_pos_weight / class_weight, NOT by
resampling inside these wrappers. SMOTE is applied externally on the training
set before calling .fit().
"""

from __future__ import annotations

import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb


# ─── Shared base ─────────────────────────────────────────────────────────────

class _BaseClassicalModel:
    """Thin wrapper enforcing the predict_proba(X) → 1-D array contract."""

    name: str = "base"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_BaseClassicalModel":
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns fraud probability (class-1 column) as a 1-D array."""
        raise NotImplementedError

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  [{self.name}] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "_BaseClassicalModel":
        return joblib.load(path)


# ─── Logistic Regression ─────────────────────────────────────────────────────

class LogisticRegressionModel(_BaseClassicalModel):
    """
    Logistic Regression baseline.

    Well-calibrated by construction (the output IS a logistic function, so
    Platt scaling is redundant — isotonic regression is used instead in Stage 3
    because the calibration improvement is still measurable at this imbalance).

    class_weight='balanced' upweights fraud examples by n_legit / n_fraud.
    """
    name = "logistic_regression"

    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self.model_ = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


# ─── Random Forest ───────────────────────────────────────────────────────────

class RandomForestModel(_BaseClassicalModel):
    """
    Random Forest.

    class_weight='balanced_subsample' recomputes class weights for each
    bootstrap sample, which is more robust than global 'balanced' for forests.

    n_estimators=300 is a reasonable default — more trees rarely hurt but
    add inference latency. min_samples_leaf=5 prevents overfitting on tiny
    fraud leaf nodes.
    """
    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 150,   # 300 → 150: halves inference time, ~same AUPRC
        max_depth: int | None = None,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


# ─── XGBoost ─────────────────────────────────────────────────────────────────

class XGBoostModel(_BaseClassicalModel):
    """
    XGBoost gradient boosted trees.

    scale_pos_weight = n_legit / n_fraud tells XGBoost the true class ratio.
    This is the recommended approach for imbalance — it changes the gradient
    magnitude for positive examples rather than artificially duplicating them.

    early_stopping_rounds=50: training stops if val AUPRC doesn't improve
    for 50 rounds. eval_set must be passed to .fit() for this to activate.
    """
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_ = None
        self.scale_pos_weight_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostModel":
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        self.scale_pos_weight_ = n_neg / max(n_pos, 1)

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight_,
            early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
            eval_metric="aucpr",
            use_label_encoder=False,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.model_.fit(
            X, y,
            eval_set=eval_set,
            verbose=False,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]


# ─── LightGBM ─────────────────────────────────────────────────────────────────

class LightGBMModel(_BaseClassicalModel):
    """
    LightGBM gradient boosted trees.

    is_unbalance=True is LightGBM's equivalent of scale_pos_weight — it
    automatically reweights classes by their inverse frequency.

    LightGBM is typically faster than XGBoost on this dataset and often
    matches or exceeds its AUPRC. Both are kept for the ensemble.
    """
    name = "lightgbm"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.model_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LightGBMModel":
        callbacks = [lgb.log_evaluation(period=-1)]  # silence per-iteration output
        if X_val is not None and y_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=False,
                )
            )

        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            is_unbalance=False,   # SMOTE handles imbalance — is_unbalance on raw data inverts predictions
            class_weight="balanced",  # lightweight additional correction on top of SMOTE
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model_.fit(
            X, y,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        import pandas as pd
        from src.data.preprocessing import get_feature_names
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=get_feature_names())
        return self.model_.predict_proba(X)[:, 1]