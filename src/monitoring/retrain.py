"""
src/monitoring/retrain.py
──────────────────────────
Retraining orchestrator — champion-challenger model lifecycle.

Retraining strategy:
────────────────────
1. Rolling window: the new model trains on the most recent N months of data,
   not the full history. Older fraud patterns that are no longer active are
   intentionally discarded. The window size is configurable.

2. Two-speed retraining:
   - Full retrain (base models + ensemble): triggered by score PSI > 0.2
     or confirmed recall dropping below threshold. Expensive, infrequent.
   - Meta-learner only retrain: triggered weekly or on mild drift (PSI 0.1–0.2).
     The base models stay fixed; only the stacking meta-learner and calibration
     are updated on recent confirmed labels. Fast, cheap.

3. Champion-Challenger:
   - The retrained model runs in SHADOW mode alongside the champion.
   - Shadow mode: the challenger scores every transaction and logs its
     decision, but the API returns the champion's decision.
   - After a configurable shadow period, if the challenger's AUPRC on
     shadow transactions (with confirmed labels) exceeds the champion's,
     the challenger is promoted to champion.
   - The old champion is archived (never deleted — always recoverable).

4. MLflow model registry:
   - Every trained model is logged to MLflow with version metadata.
   - The "Champion" alias always points to the production model.
   - The API loads by alias, so promotion = alias update, no redeployment.
"""

from __future__ import annotations

import os
import json
import time
import shutil
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from src.data.preprocessing import apply_smote, fit_and_apply, temporal_split
from src.monitoring.drift import DriftReport


MODELS_DIR    = Path("outputs/models")
ARTIFACTS_DIR = Path("outputs/artifacts")
REGISTRY_FILE = Path("outputs/models/model_registry.json")


# ─── Model registry (lightweight, file-based) ────────────────────────────────

@dataclass
class ModelRecord:
    version:      str
    path:         str
    trained_at:   str
    auprc:        float
    ece:          float
    status:       str     # "champion" | "challenger" | "shadow" | "archived"
    train_window: str     # e.g. "2024-01-01 to 2024-07-01"
    trigger:      str     # why this retrain was triggered


class ModelRegistry:
    """
    Lightweight file-based model registry.
    Mirrors the important parts of MLflow model registry for offline use.
    """

    def __init__(self, path: Path = REGISTRY_FILE):
        self.path = path
        self._records: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path) as f:
                self._records = json.load(f)
        else:
            self._records = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._records, f, indent=2)

    def register(self, record: ModelRecord) -> None:
        self._records.append({
            "version":      record.version,
            "path":         record.path,
            "trained_at":   record.trained_at,
            "auprc":        record.auprc,
            "ece":          record.ece,
            "status":       record.status,
            "train_window": record.train_window,
            "trigger":      record.trigger,
        })
        self._save()

    def get_champion(self) -> dict | None:
        for r in reversed(self._records):
            if r["status"] == "champion":
                return r
        return None

    def get_challenger(self) -> dict | None:
        for r in reversed(self._records):
            if r["status"] in ("challenger", "shadow"):
                return r
        return None

    def promote_challenger(self) -> bool:
        """
        Promotes the current challenger to champion.
        Archives the old champion.
        Returns True if promotion happened.
        """
        champion   = self.get_champion()
        challenger = self.get_challenger()

        if challenger is None:
            print("  No challenger to promote.")
            return False

        # Archive old champion
        if champion:
            for r in self._records:
                if r["version"] == champion["version"]:
                    r["status"] = "archived"

        # Promote challenger
        for r in self._records:
            if r["version"] == challenger["version"]:
                r["status"] = "champion"

        # Update production_ensemble.joblib to point to new champion
        champ_path = Path(challenger["path"])
        prod_path  = MODELS_DIR / "production_ensemble.joblib"
        if champ_path.exists():
            shutil.copy2(str(champ_path), str(prod_path))
            print(f"  Promoted {challenger['version']} → champion")
            print(f"  Copied {champ_path} → {prod_path}")

        self._save()
        return True

    def list_versions(self) -> list[dict]:
        return list(self._records)


# ─── Retraining functions ────────────────────────────────────────────────────

def retrain_meta_learner_only(
    calibrated_models: dict,
    X_recent: np.ndarray,
    y_recent: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> object:
    """
    Fast retrain: only the stacking meta-learner is updated.
    Base models and their calibrators are kept frozen.

    When to use: mild drift (PSI 0.1–0.2) or weekly scheduled update.
    The meta-learner adapts to shifts in which base models are most
    reliable on recent data, without the cost of retraining all 6 models.

    Args:
        calibrated_models: Existing { name: CalibratedModel } from Stage 3.
        X_recent, y_recent: Recent data window (preprocessed).
        X_val, y_val:       Held-out val set for meta calibration.

    Returns:
        Updated StackingEnsemble with new meta-learner.
    """
    from src.ensemble.stacking import StackingEnsemble, generate_oof_predictions
    from src.models import load_model, MODEL_NAMES

    print("  [retrain_meta] Rebuilding meta-learner on recent data...")

    # Build val matrix from calibrated models
    val_cols = []
    for name in MODEL_NAMES:
        val_cols.append(calibrated_models[name].predict_proba_calibrated(X_val))
    val_matrix = np.column_stack(val_cols).astype(np.float32)

    # Build recent OOF matrix — use calibrated models directly (no base retrain)
    recent_cols = []
    for name in MODEL_NAMES:
        recent_cols.append(calibrated_models[name].predict_proba_calibrated(X_recent))
    recent_matrix = np.column_stack(recent_cols).astype(np.float32)

    # Retrain meta-learner on recent data
    new_ensemble = StackingEnsemble(calibrated_models=calibrated_models, C=1.0)
    new_ensemble.fit_meta(
        oof_matrix = recent_matrix,
        y_train    = y_recent,
        val_matrix = val_matrix,
        y_val      = y_val,
    )
    return new_ensemble


def full_retrain(
    df_window: object,  # pd.DataFrame — rolling window of raw data
    val_frac: float = 0.15,
    random_state: int = 42,
) -> object:
    """
    Full pipeline retrain on a rolling data window.

    Runs: preprocessing → SMOTE → all base models → calibration → ensemble.
    This is the same sequence as Stages 1–4 but on the rolling window.

    Args:
        df_window:   DataFrame with raw transaction data (Time, Amount, V1–V28, Class).
        val_frac:    Fraction of window to hold out for meta-learner calibration.

    Returns:
        Newly trained StackingEnsemble.
    """
    import pandas as pd
    from src.data.preprocessing import (
        fit_and_apply, apply_smote, temporal_split, save_preprocessor
    )
    from src.models import get_all_models
    from src.calibration.selection import calibrate_all_models
    from src.ensemble.stacking import StackingEnsemble, generate_oof_predictions

    print("  [full_retrain] Starting full pipeline retrain...")
    print(f"  [full_retrain] Window size: {len(df_window):,}  "
          f"fraud: {df_window['Class'].sum()}")

    # Temporal split within the window
    train_df, val_df, _ = temporal_split(df_window, val_frac=val_frac, test_frac=0.0)

    X_train, X_val, _, y_train, y_val, _ = fit_and_apply(train_df, val_df, val_df)

    # SMOTE
    X_train_s, y_train_s = apply_smote(X_train, y_train, sampling_strategy=0.1)

    # Train base models
    models = get_all_models(input_dim=X_train.shape[1])
    for name, model in models.items():
        print(f"  [full_retrain] Training {name}...")
        if name in ("xgboost", "lightgbm", "mlp"):
            model.fit(X_train_s, y_train_s, X_val=X_val, y_val=y_val)
        elif name == "autoencoder":
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train_s, y_train_s)

    # Calibrate
    val_probs   = {name: m.predict_proba(X_val) for name, m in models.items()}
    val_logits  = {"mlp": models["mlp"].predict_logits(X_val)}
    val_logits.update({n: None for n in models if n != "mlp"})
    cal_models, _ = calibrate_all_models(models, val_probs, val_logits, y_val)

    # Build val matrix
    from src.models import MODEL_NAMES
    val_cols = [cal_models[n].predict_proba_calibrated(X_val) for n in MODEL_NAMES]
    val_matrix = np.column_stack(val_cols).astype(np.float32)

    # OOF + meta-learner
    oof = generate_oof_predictions(models, cal_models, X_train, y_train)
    ensemble = StackingEnsemble(calibrated_models=cal_models)
    ensemble.fit_meta(oof, y_train, val_matrix, y_val)

    print("  [full_retrain] Done.")
    return ensemble


# ─── Shadow mode evaluation ──────────────────────────────────────────────────

class ShadowEvaluator:
    """
    Runs a challenger model in shadow mode alongside the champion.

    The challenger scores every transaction but its decision is not returned
    to the caller — only logged internally. After the shadow period, compare
    AUPRC on confirmed labels to decide whether to promote.
    """

    def __init__(self, champion, challenger, tau_binary: float):
        self.champion   = champion
        self.challenger = challenger
        self.tau_binary = tau_binary

        self._champion_probs:   list[float] = []
        self._challenger_probs: list[float] = []
        self._true_labels:      list[int]   = []

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Scores X with both models. Returns champion probabilities.
        Challenger probabilities are logged internally.
        """
        champ_p = self.champion.predict_proba(X)
        chal_p  = self.challenger.predict_proba(X)

        self._champion_probs.extend(champ_p.tolist())
        self._challenger_probs.extend(chal_p.tolist())

        return champ_p   # API always returns champion output

    def log_confirmed_label(self, idx: int, true_label: int) -> None:
        """Log a confirmed label for transaction at position idx."""
        while len(self._true_labels) <= idx:
            self._true_labels.append(-1)
        self._true_labels[idx] = true_label

    def should_promote(self) -> tuple[bool, dict]:
        """
        Evaluates whether the challenger outperforms the champion on
        confirmed labels accumulated during the shadow period.

        Returns:
            (promote: bool, comparison: dict)
        """
        from sklearn.metrics import average_precision_score
        from src.evaluation.metrics import compute_ece

        confirmed_mask = [i for i, l in enumerate(self._true_labels) if l != -1]
        if len(confirmed_mask) < 30:
            return False, {"reason": "Not enough confirmed labels yet",
                           "n_confirmed": len(confirmed_mask)}

        y_true = np.array([self._true_labels[i] for i in confirmed_mask])
        c_probs = np.array([self._champion_probs[i]   for i in confirmed_mask])
        h_probs = np.array([self._challenger_probs[i] for i in confirmed_mask])

        champ_auprc = average_precision_score(y_true, c_probs)
        chal_auprc  = average_precision_score(y_true, h_probs)
        champ_ece   = compute_ece(y_true, c_probs)
        chal_ece    = compute_ece(y_true, h_probs)

        promote = (
            chal_auprc > champ_auprc
            or (chal_auprc == champ_auprc and chal_ece < champ_ece)
        )

        comparison = {
            "n_confirmed":     len(confirmed_mask),
            "champion_auprc":  round(champ_auprc, 6),
            "challenger_auprc":round(chal_auprc,  6),
            "champion_ece":    round(champ_ece,   6),
            "challenger_ece":  round(chal_ece,    6),
            "promote":         promote,
        }
        return promote, comparison
