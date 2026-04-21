"""
src/data/preprocessing.py
─────────────────────────
Builds and manages the preprocessing pipeline.

The pipeline is a ColumnTransformer that:
  - Encodes Time as sin/cos of hour-of-day (cyclical encoding)
  - Scales Amount with RobustScaler (outlier-robust)
  - Passes V1–V28 through unchanged (already PCA-transformed, zero-centered)

The fitted pipeline object is serialized alongside every model so that
inference always applies the exact same transforms as training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


# ─── Custom transformer ────────────────────────────────────────────────────────

class CyclicalTimeEncoder(BaseEstimator, TransformerMixin):
    """
    Converts the raw 'Time' column (seconds elapsed from dataset start) into
    two components: sin and cos of the fraction of a 24-hour day.

    This ensures the model understands that hour 23 and hour 0 are adjacent
    (distance ≈ 0) rather than maximally distant (distance = 23 hours).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        seconds = np.asarray(X).ravel()
        hour_fraction = (seconds % 86_400) / 86_400   # [0, 1)
        angle = 2.0 * np.pi * hour_fraction
        return np.column_stack([np.sin(angle), np.cos(angle)])

    def get_feature_names_out(self, input_features=None):
        return np.array(["time_sin", "time_cos"])


# ─── Pipeline builder ──────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Returns an *unfitted* ColumnTransformer.

    Output feature order:
        [time_sin, time_cos, amount_scaled, V1, V2, ..., V28]  → 31 features total
    """
    v_cols = [f"V{i}" for i in range(1, 29)]

    preprocessor = ColumnTransformer(
        transformers=[
            ("time_cyclical", CyclicalTimeEncoder(), ["Time"]),
            ("amount_scaler", RobustScaler(), ["Amount"]),
            ("v_passthrough", "passthrough", v_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def val_split(
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_a_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the validation set into two non-overlapping halves:
        val_a — used for calibration (Stage 3) and ensemble meta-learner (Stage 4)
        val_b — used ONLY for threshold optimisation (Stage 5)

    This prevents the threshold from being optimised on data that was also
    used to fit calibrators and the meta-learner, which inflates val metrics
    and causes a gap between val and test performance.

    The split is chronological (first val_a_frac rows → val_a, rest → val_b)
    to respect the temporal ordering.

    Returns:
        X_val_a, y_val_a, X_val_b, y_val_b
    """
    n = len(X_val)
    split = int(n * val_a_frac)
    return (
        X_val[:split], y_val[:split],
        X_val[split:], y_val[split:],
    )


def get_feature_names() -> list[str]:
    """Returns the ordered list of feature names produced by the pipeline."""
    return ["time_sin", "time_cos", "amount_scaled"] + [f"V{i}" for i in range(1, 29)]



def temporal_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically (by 'Time') into train / val / test.

    The test set is locked — it must only be touched once, at the end of the
    project to report final numbers. Using a random split would leak future
    temporal patterns into the training set and produce optimistic metrics
    that collapse in production.

    Returns:
        train_df, val_df, test_df  (each is a copy, reset index)
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * (1.0 - val_frac - test_frac))
    val_end   = int(n * (1.0 - test_frac))

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df   = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df  = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


# ─── Fit and apply ─────────────────────────────────────────────────────────────

FEATURE_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


def fit_and_apply(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           ColumnTransformer]:
    """
    Fits the preprocessor on train_df only, then transforms all three splits.

    IMPORTANT: The preprocessor is fitted on train_df exclusively.
    Fitting on val/test would constitute data leakage.

    Returns:
        X_train, X_val, X_test  (np.ndarray, float32)
        y_train, y_val, y_test  (np.ndarray, int)
        preprocessor            (fitted ColumnTransformer — serialize this)
    """
    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(train_df[FEATURE_COLS]).astype(np.float32)
    X_val   = preprocessor.transform(val_df[FEATURE_COLS]).astype(np.float32)
    X_test  = preprocessor.transform(test_df[FEATURE_COLS]).astype(np.float32)

    y_train = train_df["Class"].values.astype(np.int64)
    y_val   = val_df["Class"].values.astype(np.int64)
    y_test  = test_df["Class"].values.astype(np.int64)

    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor


# ─── SMOTE utility ─────────────────────────────────────────────────────────────

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE to the training set to partially address class imbalance.

    sampling_strategy=0.1 means the fraud class will be upsampled to 10% of
    the majority class size — a moderate ratio that avoids over-synthesising.

    !! ONLY call this on the training set, never on val or test. !!
    !! When using within cross-validation, apply inside each fold. !!

    Returns:
        X_resampled, y_resampled
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res.astype(np.float32), y_res.astype(np.int64)


# ─── Persistence ───────────────────────────────────────────────────────────────

def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    print(f"  Preprocessor saved → {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    return joblib.load(path)