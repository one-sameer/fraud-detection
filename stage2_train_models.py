"""
stage2_train_models.py
──────────────────────
Stage 2: Train and evaluate all six base models.

Run:
    python stage2_train_models.py

Prerequisites:
    stage1_preprocess.py must have been run first.
    outputs/artifacts/ must contain X_train.npy, X_val.npy, y_train.npy, y_val.npy

What this script does, in order:
    1. Loads preprocessed arrays from Stage 1.
    2. Applies SMOTE to the training set (externally, before model .fit()).
    3. Trains all 6 models. Gradient boosting models use X_val for early stopping.
    4. Evaluates each model on the UNMODIFIED val set (no SMOTE on val — ever).
    5. Finds the best-F1 threshold per model (exploratory — cost-aware tuning is Stage 5).
    6. Saves each model to outputs/models/.
    7. Saves each model's val-set probabilities to outputs/artifacts/ for Stage 3.
    8. Logs all metrics, plots, and model files to MLflow.
    9. Prints a final comparison table (primary sort: AUPRC).

Outputs:
    outputs/models/{name}.joblib or .pt
    outputs/artifacts/val_probs_{name}.npy   ← raw (uncalibrated) val probabilities
    outputs/artifacts/train_probs_{name}.npy ← for OOF stacking in Stage 4
    outputs/plots/pr_curves_all.png
    outputs/plots/reliability_before_calib.png
    MLflow run: stage2_train_models
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.models import get_all_models
from src.data.preprocessing import apply_smote
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
MODELS.mkdir(parents=True, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_splits() -> tuple[np.ndarray, ...]:
    print("[1/7] Loading Stage 1 artifacts...")
    arrays = {}
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        path = ARTIFACTS / f"{name}.npy"
        assert path.exists(), f"Missing {path}. Did you run stage1_preprocess.py?"
        arrays[name] = np.load(str(path))
        print(f"      {name}: {arrays[name].shape}")
    return (
        arrays["X_train"], arrays["X_val"],  arrays["X_test"],
        arrays["y_train"], arrays["y_val"],  arrays["y_test"],
    )


def _model_save_path(name: str) -> str:
    ext = ".pt" if name in ("mlp", "autoencoder") else ".joblib"
    return str(MODELS / f"{name}{ext}")


# ─── Per-model training ────────────────────────────────────────────────────────

def train_and_evaluate(
    name: str,
    model,
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_train_smote: np.ndarray,
    y_train_smote: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """
    Trains one model, evaluates on val, returns metrics dict and
    saves val_probs + train_probs to disk.
    """
    print(f"\n  ── {name} ──")

    # ── train ──
    if name == "xgboost":
        # XGBoost: raw data + scale_pos_weight handles imbalance.
        # SMOTE + scale_pos_weight double-weights fraud and hurts AUPRC.
        model.fit(X_train_raw, y_train, X_val=X_val, y_val=y_val)
    elif name == "lightgbm":
        # LightGBM: SMOTE data + is_unbalance=False (set in classical.py).
        # is_unbalance=True on raw 0.19% fraud rate causes inverted predictions
        # (AUROC < 0.5) due to bad interaction with AUCPR early stopping.
        model.fit(X_train_smote, y_train_smote, X_val=X_val, y_val=y_val)
    elif name == "mlp":
        model.fit(X_train_smote, y_train_smote, X_val=X_val, y_val=y_val)
    elif name == "autoencoder":
        model.fit(X_train_raw, y_train, X_val=X_val, y_val=y_val)
    else:
        # LR, RF: SMOTE'd data, no early stopping
        model.fit(X_train_smote, y_train_smote)

    # ── val probabilities ──
    val_probs = model.predict_proba(X_val)

    # ── train probabilities (for OOF stacking in Stage 4) ──
    X_for_train_probs = X_train_raw if name in ("autoencoder", "xgboost") else X_train_smote
    train_probs = model.predict_proba(X_for_train_probs)

    # ── metrics ──
    best_tau, best_f1 = find_best_f1_threshold(y_val, val_probs)
    metrics = compute_metrics(y_val, val_probs, threshold=best_tau)
    metrics["ece"] = compute_ece(y_val, val_probs)
    metrics["best_f1_threshold"] = round(best_tau, 4)

    print(f"      AUPRC={metrics['auprc']:.4f}  AUROC={metrics['auroc']:.4f}  "
          f"MCC={metrics['mcc']:.4f}  ECE={metrics['ece']:.4f}  "
          f"best_τ={best_tau:.3f}")

    # ── save ──
    model.save(_model_save_path(name))
    np.save(str(ARTIFACTS / f"val_probs_{name}.npy"),   val_probs)
    np.save(str(ARTIFACTS / f"train_probs_{name}.npy"), train_probs)

    return metrics


# ─── Comparison plots ─────────────────────────────────────────────────────────

def plot_all_pr_curves(
    all_val_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> plt.Figure:
    results = {name: (y_val, probs) for name, probs in all_val_probs.items()}
    return plot_pr_curve(results, title="Precision-Recall curves — all base models (val set)")


def plot_all_reliability(
    all_val_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> plt.Figure:
    """Grid of reliability diagrams for all models before calibration."""
    n = len(all_val_probs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.ravel()
    for i, (name, probs) in enumerate(all_val_probs.items()):
        plot_reliability_diagram(y_val, probs, model_name=name, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Reliability diagrams — before calibration", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 2 — BASE MODEL TRAINING")
    print("═" * 60)

    # 1. Load
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    input_dim = X_train.shape[1]

    # 2. SMOTE on train only
    print(f"\n[2/7] Applying SMOTE to training set...")
    print(f"      Before: {len(y_train):,} samples  ({y_train.sum()} fraud, "
          f"{y_train.mean()*100:.3f}%)")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, sampling_strategy=0.05)
    print(f"      After : {len(y_train_smote):,} samples  ({y_train_smote.sum()} fraud, "
          f"{y_train_smote.mean()*100:.3f}%)")

    # 3. Instantiate models
    print(f"\n[3/7] Instantiating models  (input_dim={input_dim})...")
    models = get_all_models(input_dim=input_dim)

    # 4. Train all models
    print(f"\n[4/7] Training all models...")
    all_metrics: dict[str, dict]       = {}
    all_val_probs: dict[str, np.ndarray] = {}

    for name, model in models.items():
        metrics = train_and_evaluate(
            name=name,
            model=model,
            X_train_raw=X_train,
            y_train=y_train,
            X_train_smote=X_train_smote,
            y_train_smote=y_train_smote,
            X_val=X_val,
            y_val=y_val,
        )
        all_metrics[name]    = metrics
        all_val_probs[name]  = np.load(str(ARTIFACTS / f"val_probs_{name}.npy"))

    # 5. Comparison plots
    print(f"\n[5/7] Generating comparison plots...")
    fig_pr = plot_all_pr_curves(all_val_probs, y_val)
    pr_path = PLOTS / "pr_curves_all.png"
    fig_pr.savefig(pr_path, bbox_inches="tight", dpi=130)
    plt.close(fig_pr)
    print(f"      Saved → {pr_path}")

    fig_rel = plot_all_reliability(all_val_probs, y_val)
    rel_path = PLOTS / "reliability_before_calib.png"
    fig_rel.savefig(rel_path, bbox_inches="tight", dpi=130)
    plt.close(fig_rel)
    print(f"      Saved → {rel_path}")

    # 6. Print summary table
    print(f"\n[6/7] Results summary:")
    print_metrics_table(all_metrics)

    # 7. MLflow logging
    print(f"[7/7] Logging to MLflow...")
    setup_mlflow()
    end_run_if_active()

    with mlflow.start_run(run_name="stage2_train_models"):
        mlflow.log_params({
            "n_models":          len(models),
            "smote_strategy":    0.1,
            "train_size_raw":    len(X_train),
            "train_size_smote":  len(X_train_smote),
            "val_size":          len(X_val),
        })

        for name, m in all_metrics.items():
            log_metrics_dict(m, prefix=f"{name}_")

        mlflow.log_artifact(str(pr_path),  artifact_path="plots")
        mlflow.log_artifact(str(rel_path), artifact_path="plots")

        for name in models:
            mlflow.log_artifact(_model_save_path(name), artifact_path="models")

    print("\n" + "─" * 60)
    print("  Stage 2 complete.")
    print("  Outputs:")
    print("    outputs/models/        ← 6 saved model files")
    print("    outputs/artifacts/     ← val_probs_*.npy, train_probs_*.npy")
    print("    outputs/plots/         ← pr_curves_all.png, reliability_before_calib.png")
    print("  Next step: python stage3_calibrate.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()