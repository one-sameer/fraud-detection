"""
stage5_binary_threshold.py
───────────────────────────
Stage 5: Cost-aware binary threshold optimisation.

Run:
    python stage5_binary_threshold.py

Prerequisites:
    - stage4_ensemble.py completed.
    - C_FN and C_FP set in configs/cost_config.yaml.

What this does:
    1. Validates that C_FN and C_FP are configured.
    2. Loads the ensemble's val probabilities from Stage 4.
    3. Sweeps τ ∈ [0.01, 0.99] computing ExpectedCost(τ) = C_FN×FN + C_FP×FP
       at each point on the validation set.
    4. Selects τ* = argmin ExpectedCost(τ).
    5. Plots the cost curve, PR curve at τ*, and confusion analysis.
    6. Writes τ* to configs/decision_config.yaml.
    7. Evaluates on the TEST SET (only time test set is used before Stage 6).
    8. Logs everything to MLflow.

Outputs:
    configs/decision_config.yaml        ← tau updated
    outputs/plots/cost_curve_binary.png
    outputs/plots/threshold_analysis.png
    MLflow run: stage5_binary_threshold
"""

import sys
import warnings
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.decision.threshold_optimizer import optimise_binary_threshold
from src.evaluation.metrics import compute_metrics, compute_ece
from src.utils.mlflow_utils import setup_mlflow, log_metrics_dict, end_run_if_active

ARTIFACTS = Path("outputs/artifacts")
PLOTS     = Path("outputs/plots")
CONFIG    = Path("configs/decision_config.yaml")
COST_CFG  = Path("configs/cost_config.yaml")


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_costs() -> tuple[float, float]:
    with open(COST_CFG) as f:
        cfg = yaml.safe_load(f)
    C_FN = cfg["costs"]["C_FN"]
    C_FP = cfg["costs"]["C_FP"]
    if C_FN is None or C_FP is None:
        raise ValueError(
            "\n\nC_FN and C_FP are not set in configs/cost_config.yaml.\n"
            "Set them before running this stage.\n"
            "Example values: C_FN: 125.0  C_FP: 5.0\n"
            "(C_FN = average fraud transaction amount, C_FP = operational block cost)"
        )
    return float(C_FN), float(C_FP)


def write_binary_threshold(tau: float) -> None:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["binary"]["threshold"] = round(float(tau), 4)
    with open(CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"      Written τ* = {tau:.4f} → {CONFIG}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_cost_curve(
    taus: np.ndarray,
    costs: np.ndarray,
    best_tau: float,
    best_cost: float,
    C_FN: float,
    C_FP: float,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Full cost curve
    axes[0].plot(taus, costs, color="#4A90D9", linewidth=1.8)
    axes[0].axvline(best_tau, color="#E05C5C", linestyle="--", linewidth=1.5,
                    label=f"τ* = {best_tau:.3f}  (cost = {best_cost:,.1f})")
    axes[0].scatter([best_tau], [best_cost], color="#E05C5C", s=60, zorder=5)
    axes[0].set_xlabel("Threshold τ")
    axes[0].set_ylabel("Expected cost")
    axes[0].set_title(f"Cost curve  (C_FN={C_FN:.1f}, C_FP={C_FP:.1f})")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # Zoomed view around the minimum
    zoom_lo = max(0.0, best_tau - 0.2)
    zoom_hi = min(1.0, best_tau + 0.2)
    mask = (taus >= zoom_lo) & (taus <= zoom_hi)
    axes[1].plot(taus[mask], costs[mask], color="#4A90D9", linewidth=2.0)
    axes[1].axvline(best_tau, color="#E05C5C", linestyle="--", linewidth=1.5,
                    label=f"τ* = {best_tau:.3f}")
    axes[1].scatter([best_tau], [best_cost], color="#E05C5C", s=60, zorder=5)
    axes[1].set_xlabel("Threshold τ")
    axes[1].set_ylabel("Expected cost")
    axes[1].set_title("Zoomed around minimum")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Binary cost-aware threshold optimisation", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    best_tau: float,
    C_FN: float,
    C_FP: float,
) -> plt.Figure:
    """
    3-panel figure:
      1. Precision and Recall vs threshold (shows the tradeoff)
      2. Score distribution coloured by true class
      3. FN rate and FP rate vs threshold
    """
    taus = np.linspace(0.01, 0.99, 200)
    precisions, recalls, f1s = [], [], []
    fn_rates, fp_rates       = [], []

    n_fraud = int(y_true.sum())
    n_legit = int((y_true == 0).sum())

    for tau in taus:
        pred = (y_prob >= tau).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        fn_rates.append(fn / max(n_fraud, 1))
        fp_rates.append(fp / max(n_legit, 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: Precision / Recall / F1 vs threshold
    axes[0].plot(taus, precisions, label="Precision", color="#4A90D9", linewidth=1.5)
    axes[0].plot(taus, recalls,    label="Recall",    color="#E05C5C", linewidth=1.5)
    axes[0].plot(taus, f1s,        label="F1",        color="#2ECC71", linewidth=1.5,
                 linestyle="--")
    axes[0].axvline(best_tau, color="black", linestyle=":", linewidth=1.2,
                    label=f"τ* = {best_tau:.3f}")
    axes[0].set_xlabel("Threshold τ")
    axes[0].set_title("Precision / Recall / F1 vs τ")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1)

    # Panel 2: Score distribution
    bins = np.linspace(0, 1, 60)
    axes[1].hist(y_prob[y_true == 0], bins=bins, alpha=0.55,
                 color="#4A90D9", density=True, label="Legitimate")
    axes[1].hist(y_prob[y_true == 1], bins=bins, alpha=0.70,
                 color="#E05C5C", density=True, label="Fraud")
    axes[1].axvline(best_tau, color="black", linestyle="--", linewidth=1.5,
                    label=f"τ* = {best_tau:.3f}")
    axes[1].set_xlabel("Ensemble fraud probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Score distribution by true class")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # Panel 3: FN rate and FP rate vs threshold
    axes[2].plot(taus, fn_rates, label="FN rate (missed fraud)", color="#E05C5C", linewidth=1.5)
    axes[2].plot(taus, fp_rates, label="FP rate (wrongly blocked)", color="#4A90D9",
                 linewidth=1.5)
    axes[2].axvline(best_tau, color="black", linestyle=":", linewidth=1.2,
                    label=f"τ* = {best_tau:.3f}")
    axes[2].set_xlabel("Threshold τ")
    axes[2].set_ylabel("Rate")
    axes[2].set_title("Error rates vs τ")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim(0, 1)

    fig.suptitle(f"Binary threshold analysis  (τ* = {best_tau:.3f})", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 5 — BINARY THRESHOLD OPTIMISATION")
    print("═" * 60)

    # 1. Cost config
    print("\n[1/6] Loading cost config...")
    C_FN, C_FP = load_costs()
    print(f"      C_FN = {C_FN}  |  C_FP = {C_FP}  |  ratio = {C_FN/C_FP:.1f}×")

    # 2. Load val probs + labels
    print("\n[2/6] Loading ensemble val probabilities...")
    y_val  = np.load(str(ARTIFACTS / "y_val.npy"))
    y_prob = np.load(str(ARTIFACTS / "ensemble_val_probs.npy"))
    print(f"      n_val={len(y_val):,}  fraud={y_val.sum()}  "
          f"({y_val.mean()*100:.3f}%)  "
          f"prob_range=[{y_prob.min():.4f}, {y_prob.max():.4f}]")

    # 3. Optimise threshold
    print("\n[3/6] Sweeping thresholds...")
    result, taus, costs = optimise_binary_threshold(
        y_true=y_val, y_prob=y_prob, C_FN=C_FN, C_FP=C_FP, n_steps=200
    )

    print(f"\n      ┌─────────────────────────────────────┐")
    print(f"      │  Optimal binary threshold  τ* = {result.tau:.4f}   │")
    print(f"      ├─────────────────────────────────────┤")
    print(f"      │  Expected cost  = {result.expected_cost:>10,.2f}        │")
    print(f"      │  Precision      = {result.precision:.4f}               │")
    print(f"      │  Recall         = {result.recall:.4f}               │")
    print(f"      │  F1             = {result.f1:.4f}               │")
    print(f"      │  FN count       = {result.fn_count:>5}  ({result.fn_rate*100:.2f}% of fraud) │")
    print(f"      │  FP count       = {result.fp_count:>5}  ({result.fp_rate*100:.2f}% of legit) │")
    print(f"      └─────────────────────────────────────┘")

    # 4. Write to config
    print(f"\n[4/6] Writing τ* to decision config...")
    write_binary_threshold(result.tau)

    # 5. Test set evaluation (once only)
    print(f"\n[5/6] Evaluating on TEST SET (one-time, final binary metrics)...")
    X_test   = np.load(str(ARTIFACTS / "X_test.npy"))
    y_test   = np.load(str(ARTIFACTS / "y_test.npy"))

    import joblib
    ensemble = joblib.load("outputs/models/production_ensemble.joblib")
    test_probs = ensemble.predict_proba(X_test)

    test_metrics = compute_metrics(y_test, test_probs, threshold=result.tau)
    test_metrics["ece"] = compute_ece(y_test, test_probs)

    print(f"\n      ── Test set results at τ* = {result.tau:.4f} ──")
    print(f"      AUPRC     = {test_metrics['auprc']:.6f}")
    print(f"      AUROC     = {test_metrics['auroc']:.6f}")
    print(f"      MCC       = {test_metrics['mcc']:.6f}")
    print(f"      F1        = {test_metrics['f1']:.6f}")
    print(f"      Precision = {test_metrics['precision']:.6f}")
    print(f"      Recall    = {test_metrics['recall']:.6f}")
    print(f"      ECE       = {test_metrics['ece']:.6f}")
    print(f"      TP={test_metrics['tp']}  FP={test_metrics['fp']}  "
          f"FN={test_metrics['fn']}  TN={test_metrics['tn']}")

    np.save(str(ARTIFACTS / "test_probs.npy"), test_probs)

    # 6. Plots
    print(f"\n[6/6] Saving plots...")
    fig_cost = plot_cost_curve(taus, costs, result.tau, result.expected_cost, C_FN, C_FP)
    fig_cost.savefig(PLOTS / "cost_curve_binary.png", bbox_inches="tight", dpi=130)
    plt.close(fig_cost)

    fig_analysis = plot_threshold_analysis(y_val, y_prob, result.tau, C_FN, C_FP)
    fig_analysis.savefig(PLOTS / "threshold_analysis_binary.png", bbox_inches="tight", dpi=130)
    plt.close(fig_analysis)
    print(f"      Saved → outputs/plots/cost_curve_binary.png")
    print(f"      Saved → outputs/plots/threshold_analysis_binary.png")

    # MLflow
    setup_mlflow()
    end_run_if_active()
    with mlflow.start_run(run_name="stage5_binary_threshold"):
        mlflow.log_params({"C_FN": C_FN, "C_FP": C_FP, "tau_star": result.tau})
        mlflow.log_metrics({
            "val_expected_cost": result.expected_cost,
            "val_fn_rate":       result.fn_rate,
            "val_fp_rate":       result.fp_rate,
        })
        log_metrics_dict(test_metrics, prefix="test_")
        mlflow.log_artifact(str(PLOTS / "cost_curve_binary.png"),          "plots")
        mlflow.log_artifact(str(PLOTS / "threshold_analysis_binary.png"),  "plots")
        mlflow.log_artifact(str(CONFIG), "configs")

    print("\n" + "─" * 60)
    print("  Stage 5 complete.")
    print(f"  τ* = {result.tau:.4f}  written to configs/decision_config.yaml")
    print("  Next step: python stage6_triclass_threshold.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()