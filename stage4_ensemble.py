"""
stage4_ensemble.py
───────────────────
Stage 4: Build and evaluate the stacking ensemble.

Run:
    python stage4_ensemble.py

Prerequisites:
    stage3_calibrate.py must have been run first.

What this script does, in order:
    1.  Loads all splits (X_train, X_val, y_train, y_val).
    2.  Loads all 6 calibrated models from Stage 3.
    3.  Loads the pre-computed calibrated val probabilities from Stage 3.
    4.  Builds the val feature matrix (n_val × 6) from Stage 3 cal probs.
    5.  Generates the OOF matrix (n_train × 6) by retraining each base model
        on each of 5 CV folds and applying the Stage 3 calibrators.
        [Most time-consuming step — ~same runtime as Stage 2 × 5 folds]
    6.  Trains the stacking meta-learner (LR) on the OOF matrix.
    7.  Calibrates the meta-learner on the val set.
    8.  Trains the weighted average ensemble on the val matrix (fast).
    9.  Evaluates both ensembles on the val set.
    10. Selects the winner (higher AUPRC + lower ECE).
    11. Saves both ensembles and the winner pointer.
    12. Saves winner val probabilities → outputs/artifacts/ensemble_val_probs.npy
    13. Logs everything to MLflow.

Outputs:
    outputs/models/stacking_ensemble.joblib
    outputs/models/weighted_avg_ensemble.joblib
    outputs/models/production_ensemble.joblib   ← symlink to winner
    outputs/artifacts/ensemble_val_probs.npy    ← winner's val probs (→ Stage 5)
    outputs/artifacts/oof_matrix.npy            ← OOF matrix (audit / debugging)
    outputs/plots/ensemble_comparison.png
    outputs/plots/reliability_ensemble.png
    MLflow run: stage4_ensemble
"""

import sys
import warnings
import joblib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.models import load_model, MODEL_NAMES, ENSEMBLE_MODEL_NAMES
from src.ensemble.stacking import StackingEnsemble, generate_oof_predictions
from src.ensemble.weighted_avg import WeightedAverageEnsemble
from src.evaluation.metrics import (
    compute_metrics,
    compute_ece,
    find_best_f1_threshold,
    plot_reliability_diagram,
    plot_pr_curve,
    print_metrics_table,
)
from src.utils.mlflow_utils import (
    setup_mlflow,
    log_figure,
    log_metrics_dict,
    end_run_if_active,
)

ARTIFACTS = Path("outputs/artifacts")
MODELS    = Path("outputs/models")
PLOTS     = Path("outputs/plots")


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_calibrated_models() -> dict:
    models = {}
    for name in MODEL_NAMES:
        path = MODELS / f"calibrated_{name}.joblib"
        assert path.exists(), f"Missing {path}. Run stage3 first."
        models[name] = joblib.load(str(path))
        print(f"      Loaded calibrated_{name}")
    return models


def load_base_models() -> dict:
    """Load the original (non-calibrated) base models — needed for OOF retraining."""
    models = {}
    for name in MODEL_NAMES:
        ext  = ".pt" if name in ("mlp", "autoencoder") else ".joblib"
        path = MODELS / f"{name}{ext}"
        assert path.exists(), f"Missing {path}. Run stage2 first."
        models[name] = load_model(name, str(path))
    return models


def build_val_matrix(calibrated_models: dict, X_val: np.ndarray) -> np.ndarray:
    """
    Builds the (n_val × n_ensemble_models) matrix from pre-saved Stage 3 probs.
    Autoencoder excluded — only ENSEMBLE_MODEL_NAMES used.
    """
    cols = []
    for name in ENSEMBLE_MODEL_NAMES:
        path = ARTIFACTS / f"cal_val_probs_{name}.npy"
        cols.append(np.load(str(path)))
    matrix = np.column_stack(cols).astype(np.float32)
    print(f"      Val feature matrix: {matrix.shape}  (models: {list(ENSEMBLE_MODEL_NAMES)})")
    return matrix


# ─── Comparison plots ─────────────────────────────────────────────────────────

def plot_ensemble_comparison(
    stacking_probs: np.ndarray,
    wavg_probs: np.ndarray,
    base_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> plt.Figure:
    """PR curves: both ensembles + all base models for context."""
    results = {
        "★ Stacking ensemble":        (y_val, stacking_probs),
        "★ Weighted avg ensemble":    (y_val, wavg_probs),
    }
    for name, probs in base_probs.items():
        results[f"  {name}"] = (y_val, probs)
    return plot_pr_curve(results, title="PR curves — ensembles vs base models (val set)")


def plot_score_distributions(
    stacking_probs: np.ndarray,
    wavg_probs: np.ndarray,
    y_val: np.ndarray,
) -> plt.Figure:
    """
    Histogram of ensemble output probabilities split by true class.
    A well-separated distribution (fraud probs high, legit probs low) is
    exactly what we want before threshold tuning in Stage 5.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, probs, title in [
        (axes[0], stacking_probs, "Stacking ensemble"),
        (axes[1], wavg_probs,     "Weighted avg ensemble"),
    ]:
        legit_p = probs[y_val == 0]
        fraud_p = probs[y_val == 1]
        bins = np.linspace(0, 1, 60)
        ax.hist(legit_p, bins=bins, alpha=0.6, color="#4A90D9",
                label=f"Legitimate  (n={len(legit_p):,})", density=True)
        ax.hist(fraud_p, bins=bins, alpha=0.7, color="#E05C5C",
                label=f"Fraud  (n={len(fraud_p):,})", density=True)
        ax.set_xlabel("Ensemble fraud probability")
        ax.set_ylabel("Density")
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-0.02, 1.02)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Score distributions — separation between classes determines threshold quality",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    return fig


def plot_meta_coefs(coefs: dict) -> plt.Figure:
    """Bar chart of stacking meta-learner coefficients per base model."""
    names  = list(coefs.keys())
    values = list(coefs.values())
    colors = ["#4A90D9" if v >= 0 else "#E05C5C" for v in values]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, values, color=colors, edgecolor="white", width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Meta-learner coefficient")
    ax.set_title("Stacking meta-learner: learned weights per base model", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return fig


def print_ensemble_comparison(
    stacking_metrics: dict,
    wavg_metrics: dict,
    winner: str,
) -> None:
    print("\n" + "─" * 65)
    print(f"{'':25} {'Stacking':>15} {'Weighted avg':>15}")
    print("─" * 65)
    for key in ["auprc", "auroc", "mcc", "f1", "recall", "ece"]:
        sm = stacking_metrics.get(key, float("nan"))
        wm = wavg_metrics.get(key, float("nan"))
        better_s = " ◄" if (key != "ece" and sm > wm) or (key == "ece" and sm < wm) else ""
        better_w = " ◄" if (key != "ece" and wm > sm) or (key == "ece" and wm < sm) else ""
        print(f"  {key:<23} {sm:>14.6f}{better_s:<3} {wm:>14.6f}{better_w}")
    print("─" * 65)
    print(f"  WINNER: {winner}")
    print("─" * 65 + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 4 — STACKING ENSEMBLE")
    print("═" * 60)

    # 1. Load data
    print("\n[1/8] Loading data splits...")
    X_train = np.load(str(ARTIFACTS / "X_train.npy"))
    y_train = np.load(str(ARTIFACTS / "y_train.npy"))
    X_val   = np.load(str(ARTIFACTS / "X_val.npy"))
    y_val   = np.load(str(ARTIFACTS / "y_val.npy"))
    print(f"      X_train={X_train.shape}  X_val={X_val.shape}")

    # 2. Load calibrated + base models
    print("\n[2/8] Loading calibrated models (Stage 3)...")
    calibrated_models = load_calibrated_models()

    print("\n[3/8] Loading base models (needed for OOF retraining)...")
    base_models = load_base_models()

    # 3. Build val feature matrix from pre-saved Stage 3 probs
    print("\n[4/8] Building val feature matrix from Stage 3 calibrated probs...")
    val_matrix = build_val_matrix(calibrated_models, X_val)

    # 4. OOF matrix — the expensive step
    print(f"\n[5/8] Generating OOF predictions (5-fold CV on training set)...")
    print(f"      This retrains all 6 models × 5 folds — takes a while...")
    oof_matrix = generate_oof_predictions(
        base_models       = base_models,
        calibrated_models = calibrated_models,
        X_train           = X_train,
        y_train           = y_train,
        n_splits          = 5,
    )
    np.save(str(ARTIFACTS / "oof_matrix.npy"), oof_matrix)
    print(f"\n      OOF matrix saved: {oof_matrix.shape}")
    print(f"      OOF fraud mean probs: "
          + "  ".join(f"{oof_matrix[y_train==1, i].mean():.4f}"
                      for i in range(oof_matrix.shape[1])))

    # 5. Stacking ensemble
    print(f"\n[6/8] Training stacking ensemble...")
    stacking = StackingEnsemble(
        calibrated_models=calibrated_models,
        C=1.0,
    )
    stacking.fit_meta(
        oof_matrix = oof_matrix,
        y_train    = y_train,
        val_matrix = val_matrix,
        y_val      = y_val,
    )
    stacking_val_probs = stacking.meta_calibrator_.predict(
        stacking.meta_learner_.predict_proba(val_matrix)[:, 1]
    )

    # 6. Weighted average ensemble
    print(f"\n[7/8] Training weighted average ensemble...")
    wavg = WeightedAverageEnsemble(calibrated_models=calibrated_models)
    wavg.fit(val_matrix=val_matrix, y_val=y_val)
    wavg_val_probs = wavg.meta_calibrator_.predict(
        (val_matrix * wavg.weights_).sum(axis=1)
    )

    # 7. Evaluate both
    print(f"\n[8/8] Evaluating and selecting winner...")
    stacking_tau, _ = find_best_f1_threshold(y_val, stacking_val_probs)
    wavg_tau, _     = find_best_f1_threshold(y_val, wavg_val_probs)

    stacking_metrics = compute_metrics(y_val, stacking_val_probs, threshold=stacking_tau)
    stacking_metrics["ece"] = compute_ece(y_val, stacking_val_probs)

    wavg_metrics = compute_metrics(y_val, wavg_val_probs, threshold=wavg_tau)
    wavg_metrics["ece"] = compute_ece(y_val, wavg_val_probs)

    # Winner: higher AUPRC is primary criterion; ECE as tiebreaker
    if stacking_metrics["auprc"] > wavg_metrics["auprc"]:
        winner_name  = "stacking"
        winner       = stacking
        winner_probs = stacking_val_probs
    elif wavg_metrics["auprc"] > stacking_metrics["auprc"]:
        winner_name  = "weighted_avg"
        winner       = wavg
        winner_probs = wavg_val_probs
    else:
        # AUPRC tie — prefer lower ECE
        if stacking_metrics["ece"] <= wavg_metrics["ece"]:
            winner_name, winner, winner_probs = "stacking",    stacking, stacking_val_probs
        else:
            winner_name, winner, winner_probs = "weighted_avg", wavg,    wavg_val_probs

    print_ensemble_comparison(stacking_metrics, wavg_metrics, winner_name)

    # Save models
    stacking.save(str(MODELS / "stacking_ensemble.joblib"))
    wavg.save(str(MODELS / "weighted_avg_ensemble.joblib"))
    winner.save(str(MODELS / "production_ensemble.joblib"))
    np.save(str(ARTIFACTS / "ensemble_val_probs.npy"), winner_probs)
    print(f"\n      Winner '{winner_name}' saved as production_ensemble.joblib")
    print(f"      Winner val probs saved → outputs/artifacts/ensemble_val_probs.npy")

    # Plots
    base_cal_probs = {
        name: np.load(str(ARTIFACTS / f"cal_val_probs_{name}.npy"))
        for name in ENSEMBLE_MODEL_NAMES
    }

    fig_cmp = plot_ensemble_comparison(stacking_val_probs, wavg_val_probs, base_cal_probs, y_val)
    fig_cmp.savefig(PLOTS / "ensemble_comparison.png", bbox_inches="tight", dpi=130)
    plt.close(fig_cmp)

    fig_dist = plot_score_distributions(stacking_val_probs, wavg_val_probs, y_val)
    fig_dist.savefig(PLOTS / "ensemble_score_distributions.png", bbox_inches="tight", dpi=130)
    plt.close(fig_dist)

    fig_rel = plt.figure(figsize=(5, 5))
    ax = fig_rel.add_subplot(111)
    plot_reliability_diagram(y_val, winner_probs,
                             model_name=f"Production ensemble ({winner_name})", ax=ax)
    fig_rel.savefig(PLOTS / "reliability_ensemble.png", bbox_inches="tight", dpi=130)
    plt.close(fig_rel)

    if hasattr(stacking, "meta_coefs_") and stacking.meta_coefs_:
        fig_coef = plot_meta_coefs(stacking.meta_coefs_)
        fig_coef.savefig(PLOTS / "meta_learner_coefs.png", bbox_inches="tight", dpi=130)
        plt.close(fig_coef)

    # MLflow
    setup_mlflow()
    end_run_if_active()

    with mlflow.start_run(run_name="stage4_ensemble"):
        mlflow.log_param("winner", winner_name)
        log_metrics_dict(stacking_metrics, prefix="stacking_")
        log_metrics_dict(wavg_metrics,     prefix="wavg_")

        if stacking.meta_coefs_:
            for name, coef in stacking.meta_coefs_.items():
                mlflow.log_metric(f"meta_coef_{name}", coef)
        for w_name, w_val in zip(MODEL_NAMES, wavg.weights_):
            mlflow.log_metric(f"wavg_weight_{w_name}", float(w_val))

        for fname in [
            "ensemble_comparison.png",
            "ensemble_score_distributions.png",
            "reliability_ensemble.png",
            "meta_learner_coefs.png",
        ]:
            p = PLOTS / fname
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="plots")

        for fname in [
            "stacking_ensemble.joblib",
            "weighted_avg_ensemble.joblib",
            "production_ensemble.joblib",
        ]:
            mlflow.log_artifact(str(MODELS / fname), artifact_path="ensemble_models")

    print("\n" + "─" * 60)
    print("  Stage 4 complete.")
    print("  Outputs:")
    print("    outputs/models/production_ensemble.joblib  ← winner")
    print("    outputs/artifacts/ensemble_val_probs.npy   ← winner val probs")
    print("    outputs/plots/ensemble_*.png               ← comparison plots")
    print("  Next step: python stage5_binary_threshold.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()