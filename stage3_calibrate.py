"""
stage3_calibrate.py
────────────────────
Stage 3: Probability calibration for all base models.

Run:
    python stage3_calibrate.py

Prerequisites:
    stage2_train_models.py must have been run first.

What this script does, in order:
    1. Loads all 6 fitted models from outputs/models/.
    2. Loads val_probs_*.npy from outputs/artifacts/.
    3. Generates val logits for MLP (needed for temperature scaling).
    4. Calibrates each model using the chosen strategy:
           LR, RF, XGB, LightGBM, Autoencoder → cross-validated isotonic regression
           MLP                                 → temperature scaling on logits
    5. Measures ECE before and after calibration on the validation set.
    6. Saves calibrated val probabilities → outputs/artifacts/cal_val_probs_{name}.npy
    7. Saves each CalibratedModel → outputs/models/calibrated_{name}.joblib
    8. Produces side-by-side reliability diagrams (before vs after per model).
    9. Logs everything to MLflow.

Outputs:
    outputs/models/calibrated_{name}.joblib        ← CalibratedModel objects
    outputs/artifacts/cal_val_probs_{name}.npy     ← calibrated val probs (→ Stage 4)
    outputs/plots/reliability_after_calib.png
    outputs/plots/reliability_comparison_{name}.png  (one per model)
    MLflow run: stage3_calibrate
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

from src.models import load_model, MODEL_NAMES
from src.calibration.selection import calibrate_all_models, CALIBRATION_STRATEGY
from src.evaluation.metrics import (
    compute_ece,
    plot_reliability_diagram,
    print_metrics_table,
    compute_metrics,
    find_best_f1_threshold,
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

def load_base_models() -> dict:
    models = {}
    for name in MODEL_NAMES:
        ext  = ".pt" if name in ("mlp", "autoencoder") else ".joblib"
        path = str(MODELS / f"{name}{ext}")
        assert Path(path).exists(), f"Missing {path}. Run stage2 first."
        models[name] = load_model(name, path)
        print(f"      Loaded {name} ← {path}")
    return models


def load_val_probs() -> dict[str, np.ndarray]:
    probs = {}
    for name in MODEL_NAMES:
        path = ARTIFACTS / f"val_probs_{name}.npy"
        assert path.exists(), f"Missing {path}. Run stage2 first."
        probs[name] = np.load(str(path))
    return probs


def get_val_logits(models: dict, X_val: np.ndarray) -> dict[str, np.ndarray | None]:
    """
    MLP temperature scaling needs raw logits, not probabilities.
    All other models return None.
    """
    logits = {}
    for name in MODEL_NAMES:
        if name == "mlp":
            print(f"      Generating logits for MLP temperature scaling...")
            logits[name] = models[name].predict_logits(X_val)
        else:
            logits[name] = None
    return logits


# ─── Reliability comparison plots ─────────────────────────────────────────────

def plot_before_after(
    name: str,
    y_val: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    strategy: str,
) -> plt.Figure:
    """
    Side-by-side reliability diagram for one model before vs after calibration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    ece_before = compute_ece(y_val, raw_probs)
    ece_after  = compute_ece(y_val, cal_probs)

    plot_reliability_diagram(y_val, raw_probs,
                             model_name=f"{name} (raw, ECE={ece_before:.5f})",
                             ax=axes[0])
    plot_reliability_diagram(y_val, cal_probs,
                             model_name=f"{name} ({strategy}, ECE={ece_after:.5f})",
                             ax=axes[1])

    axes[0].set_title("Before calibration", fontsize=11)
    axes[1].set_title("After calibration",  fontsize=11)

    improvement = ece_before - ece_after
    colour = "green" if improvement > 0 else "red"
    fig.suptitle(
        f"{name}  —  ECE: {ece_before:.5f} → {ece_after:.5f}  "
        f"(Δ = {improvement:+.5f})",
        fontsize=12, color=colour, y=1.01,
    )
    fig.tight_layout()
    return fig


def plot_all_after_calib(
    all_cal_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> plt.Figure:
    """Grid of reliability diagrams for all models after calibration."""
    n     = len(all_cal_probs)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = axes.ravel()
    for i, (name, probs) in enumerate(all_cal_probs.items()):
        plot_reliability_diagram(y_val, probs, model_name=name, ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Reliability diagrams — after calibration", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def print_calibration_table(cal_metrics: dict[str, dict]) -> None:
    header = (f"{'Model':<25} {'Method':<12} "
              f"{'ECE before':>11} {'ECE after':>10} {'Δ ECE':>10}")
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name, m in cal_metrics.items():
        delta = m["ece_improvement"]
        mark  = "▼" if delta > 0 else "▲"   # ▼ = improved (lower ECE)
        print(
            f"{name:<25} {m['calibration_method']:<12} "
            f"{m['ece_before']:>11.6f} {m['ece_after']:>10.6f} "
            f"{mark}{abs(delta):>9.6f}"
        )
    print("─" * len(header) + "\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 3 — PROBABILITY CALIBRATION")
    print("═" * 60)

    # 1. Load data
    print("\n[1/7] Loading val split...")
    X_val = np.load(str(ARTIFACTS / "X_val.npy"))
    y_val = np.load(str(ARTIFACTS / "y_val.npy"))
    print(f"      X_val: {X_val.shape}  |  fraud: {y_val.sum()} ({y_val.mean()*100:.3f}%)")

    # 2. Load base models
    print("\n[2/7] Loading base models...")
    base_models = load_base_models()

    # 3. Load raw val probs (generated by Stage 2, no re-inference needed)
    print("\n[3/7] Loading raw val probabilities from Stage 2...")
    val_probs = load_val_probs()

    # 4. Logits only needed if any model uses temperature scaling — currently none do
    print("\n[4/7] Preparing val inputs for calibration...")
    val_logits = {name: None for name in MODEL_NAMES}

    # 5. Calibrate
    print("\n[5/7] Calibrating all models...")
    calibrated_models, cal_metrics = calibrate_all_models(
        base_models = base_models,
        val_probs   = val_probs,
        val_logits  = val_logits,
        y_val       = y_val,
    )

    # 6. Collect calibrated val probabilities — all models use isotonic now
    all_cal_probs: dict[str, np.ndarray] = {}
    for name, cal_model in calibrated_models.items():
        cal_probs = cal_model.calibrator_.predict(val_probs[name])
        all_cal_probs[name] = cal_probs
        np.save(str(ARTIFACTS / f"cal_val_probs_{name}.npy"), cal_probs)

    # 7. Save calibrated models
    print("\n[6/7] Saving calibrated models...")
    for name, cal_model in calibrated_models.items():
        path = str(MODELS / f"calibrated_{name}.joblib")
        joblib.dump(cal_model, path)
        print(f"      Saved → {path}")

    # 8. Plots
    print("\n[7/7] Generating calibration plots...")

    # per-model before/after
    per_model_figs = {}
    for name in MODEL_NAMES:
        fig = plot_before_after(
            name      = name,
            y_val     = y_val,
            raw_probs = val_probs[name],
            cal_probs = all_cal_probs[name],
            strategy  = CALIBRATION_STRATEGY[name],
        )
        path = PLOTS / f"reliability_comparison_{name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=130)
        plt.close(fig)
        per_model_figs[name] = path
        print(f"      Saved → {path}")

    # all-after grid
    fig_after = plot_all_after_calib(all_cal_probs, y_val)
    after_path = PLOTS / "reliability_after_calib.png"
    fig_after.savefig(after_path, bbox_inches="tight", dpi=130)
    plt.close(fig_after)
    print(f"      Saved → {after_path}")

    # 9. Print calibration table
    print_calibration_table(cal_metrics)

    # 10. Post-calibration metrics (AUPRC unchanged since calibration is monotone,
    #     but ECE and reliability are what matter here)
    print("Post-calibration val metrics (calibrated probabilities):")
    post_metrics = {}
    for name, cal_probs in all_cal_probs.items():
        best_tau, _ = find_best_f1_threshold(y_val, cal_probs)
        m = compute_metrics(y_val, cal_probs, threshold=best_tau)
        m["ece"] = compute_ece(y_val, cal_probs)
        post_metrics[name] = m

    print_metrics_table(post_metrics)

    # 11. MLflow
    setup_mlflow()
    end_run_if_active()

    with mlflow.start_run(run_name="stage3_calibrate"):
        for name, m in cal_metrics.items():
            mlflow.log_metrics({
                f"{name}_ece_before":     m["ece_before"],
                f"{name}_ece_after":      m["ece_after"],
                f"{name}_ece_improvement": m["ece_improvement"],
            })
        for name, m in post_metrics.items():
            log_metrics_dict(m, prefix=f"{name}_cal_")

        mlflow.log_artifact(str(after_path), artifact_path="plots")
        for name, fig_path in per_model_figs.items():
            mlflow.log_artifact(str(fig_path), artifact_path="plots/per_model")
        for name in MODEL_NAMES:
            mlflow.log_artifact(
                str(MODELS / f"calibrated_{name}.joblib"),
                artifact_path="calibrated_models",
            )

    print("\n" + "─" * 60)
    print("  Stage 3 complete.")
    print("  Outputs:")
    print("    outputs/models/calibrated_*.joblib  ← CalibratedModel objects")
    print("    outputs/artifacts/cal_val_probs_*   ← calibrated val probabilities")
    print("    outputs/plots/reliability_*.png     ← before/after diagrams")
    print("  Next step: python stage4_ensemble.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()