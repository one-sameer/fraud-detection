"""
stage6_triclass_threshold.py
─────────────────────────────
Stage 6: Cost-aware 3-class threshold optimisation.

Run:
    python stage6_triclass_threshold.py

Prerequisites:
    - stage5_binary_threshold.py completed.
    - τ* written to configs/decision_config.yaml.
    - All cost params set in configs/cost_config.yaml.

What this does:
    1. Loads τ* from Stage 5 (anchors the search space).
    2. Loads all cost parameters from cost_config.yaml.
    3. Runs joint grid search over (τ_low, τ_high):
           τ_low  ∈ [0.01, τ* - 0.01]
           τ_high ∈ [τ* + 0.01, 0.99]
       minimising:
           C_FN × escaped_fraud
         + C_FP × wrongly_blocked
         + C_review × n_reviewed
         + C_overflow × max(0, review_rate - R_max)
    4. Reports full 3-class metrics at (τ_low*, τ_high*).
    5. Runs sensitivity analysis: how do the thresholds shift as
       C_FN/C_FP ratio and R_max change? This is the key diagnostic
       for understanding how robust the solution is to cost assumptions.
    6. Writes τ_low*, τ_high* to configs/decision_config.yaml.
    7. Evaluates on the test set.
    8. Logs everything to MLflow.

Outputs:
    configs/decision_config.yaml           ← tau_low, tau_high updated
    outputs/plots/cost_grid_triclass.png   ← 2-D cost heatmap
    outputs/plots/triclass_analysis.png    ← score distributions with both thresholds
    outputs/plots/sensitivity_analysis.png ← threshold stability under cost changes
    MLflow run: stage6_triclass_threshold
"""

import sys
import warnings
import yaml
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.decision.threshold_optimizer import (
    optimise_triclass_thresholds,
    classify_transaction,
    TriClassThresholdResult,
)
from src.evaluation.metrics import compute_metrics, compute_ece
from src.utils.mlflow_utils import setup_mlflow, log_metrics_dict, end_run_if_active

ARTIFACTS = Path("outputs/artifacts")
MODELS    = Path("outputs/models")
PLOTS     = Path("outputs/plots")
CONFIG    = Path("configs/decision_config.yaml")
COST_CFG  = Path("configs/cost_config.yaml")


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_all_costs() -> dict:
    with open(COST_CFG) as f:
        cfg = yaml.safe_load(f)
    costs = cfg["costs"]
    review = cfg["review"]
    if costs["C_FN"] is None or costs["C_FP"] is None:
        raise ValueError("C_FN and C_FP must be set in configs/cost_config.yaml")
    return {
        "C_FN":      float(costs["C_FN"]),
        "C_FP":      float(costs["C_FP"]),
        "C_review":  float(costs["C_review"]),
        "C_overflow":float(costs["C_overflow"]),
        "R_max":     float(review["R_max"]),
    }


def load_binary_threshold() -> float:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    tau = cfg["binary"]["threshold"]
    if tau is None:
        raise ValueError("Binary threshold not set. Run stage5 first.")
    return float(tau)


def write_triclass_thresholds(tau_low: float, tau_high: float) -> None:
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["three_class"]["tau_low"]  = round(float(tau_low), 4)
    cfg["three_class"]["tau_high"] = round(float(tau_high), 4)
    with open(CONFIG, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"      Written τ_low={tau_low:.4f}, τ_high={tau_high:.4f} → {CONFIG}")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_cost_grid(
    cost_grid: np.ndarray,
    low_grid: np.ndarray,
    high_grid: np.ndarray,
    best_tau_low: float,
    best_tau_high: float,
    tau_binary: float,
) -> plt.Figure:
    """2-D heatmap of composite costs over the (τ_low, τ_high) search space."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Mask inf values for display
    display = np.where(np.isinf(cost_grid), np.nan, cost_grid)
    im = ax.imshow(
        display,
        origin="lower",
        aspect="auto",
        extent=[high_grid[0], high_grid[-1], low_grid[0], low_grid[-1]],
        cmap="RdYlGn_r",
    )
    plt.colorbar(im, ax=ax, label="Composite cost")

    # Mark optimum
    ax.scatter([best_tau_high], [best_tau_low], color="white", s=120,
               zorder=5, marker="*", label=f"Optimum ({best_tau_low:.3f}, {best_tau_high:.3f})")

    # Mark binary threshold contour
    ax.axhline(tau_binary, color="cyan", linestyle="--", linewidth=1.0,
               label=f"τ_binary = {tau_binary:.3f}")
    ax.axvline(tau_binary, color="cyan", linestyle="--", linewidth=1.0)

    ax.set_xlabel("τ_high")
    ax.set_ylabel("τ_low")
    ax.set_title("Composite cost grid — 3-class threshold search", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_triclass_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tau_low: float,
    tau_high: float,
    tau_binary: float,
    result: TriClassThresholdResult,
) -> plt.Figure:
    """
    Score distribution with all three decision zones marked.
    This is the diagnostic plot that shows whether the review band
    is capturing genuinely ambiguous cases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Score distribution with zones
    bins = np.linspace(0, 1, 80)
    ax  = axes[0]
    ax.hist(y_prob[y_true == 0], bins=bins, alpha=0.5, color="#4A90D9",
            density=True, label="Legitimate")
    ax.hist(y_prob[y_true == 1], bins=bins, alpha=0.7, color="#E05C5C",
            density=True, label="Fraud")

    ax.axvline(tau_low,    color="#2ECC71", linewidth=2.0, linestyle="--",
               label=f"τ_low = {tau_low:.3f}")
    ax.axvline(tau_high,   color="#E74C3C", linewidth=2.0, linestyle="--",
               label=f"τ_high = {tau_high:.3f}")
    ax.axvline(tau_binary, color="black",   linewidth=1.0, linestyle=":",
               label=f"τ_binary = {tau_binary:.3f}")

    ymax = ax.get_ylim()[1]
    ax.axvspan(0,         tau_low,  alpha=0.08, color="#2ECC71", label="Auto-approve zone")
    ax.axvspan(tau_low,   tau_high, alpha=0.08, color="#F39C12", label="Review zone")
    ax.axvspan(tau_high,  1.0,      alpha=0.08, color="#E74C3C", label="Auto-block zone")

    ax.set_xlabel("Ensemble fraud probability")
    ax.set_ylabel("Density")
    ax.set_title("Score distribution with decision zones")
    ax.legend(fontsize=7.5, loc="upper center")
    ax.grid(alpha=0.25)

    # Panel 2: Class breakdown table
    n_total  = len(y_true)
    n_fraud  = int(y_true.sum())
    n_legit  = n_total - n_fraud

    auto_approve = y_prob < tau_low
    auto_block   = y_prob > tau_high
    review_mask  = (~auto_approve) & (~auto_block)

    rows = [
        ["",                   "Fraud cases",       "Legit cases",       "Total"],
        ["Auto-approve",
         f"{((y_true==1) & auto_approve).sum()} ({((y_true==1)&auto_approve).sum()/n_fraud*100:.1f}%)",
         f"{((y_true==0) & auto_approve).sum()} ({((y_true==0)&auto_approve).sum()/n_legit*100:.1f}%)",
         f"{auto_approve.sum()}"],
        ["Human review",
         f"{((y_true==1) & review_mask).sum()} ({((y_true==1)&review_mask).sum()/n_fraud*100:.1f}%)",
         f"{((y_true==0) & review_mask).sum()} ({((y_true==0)&review_mask).sum()/n_legit*100:.1f}%)",
         f"{review_mask.sum()}  ({result.review_rate*100:.2f}%)"],
        ["Auto-block",
         f"{((y_true==1) & auto_block).sum()} ({((y_true==1)&auto_block).sum()/n_fraud*100:.1f}%)",
         f"{((y_true==0) & auto_block).sum()} ({((y_true==0)&auto_block).sum()/n_legit*100:.1f}%)",
         f"{auto_block.sum()}"],
    ]

    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 2.0)
    for j in range(4):
        tbl[(0, j)].set_facecolor("#2C2C2A")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    tbl[(1, 0)].set_facecolor("#D5F5E3")   # approve → green
    tbl[(2, 0)].set_facecolor("#FDEBD0")   # review  → orange
    tbl[(3, 0)].set_facecolor("#FADBD8")   # block   → red
    axes[1].set_title(
        f"Decision breakdown  (τ_low={tau_low:.3f}, τ_high={tau_high:.3f})",
        fontsize=11
    )

    fig.suptitle("3-class decision analysis", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_sensitivity_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    base_costs: dict,
    tau_binary: float,
) -> plt.Figure:
    """
    Shows how τ_low and τ_high shift as the C_FN/C_FP ratio and R_max change.
    This is the critical diagnostic for cost assumption robustness:
    if the thresholds barely move under reasonable cost variations, the
    solution is stable. If they jump around, the exact cost values matter a lot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Vary C_FN/C_FP ratio ──
    ratios   = [5, 10, 20, 50, 100, 200, 500]
    tau_lows, tau_highs = [], []
    C_FP = base_costs["C_FP"]
    for ratio in ratios:
        C_FN_test = C_FP * ratio
        res, _ = optimise_triclass_thresholds(
            y_true, y_prob,
            C_FN=C_FN_test, C_FP=C_FP,
            C_review=base_costs["C_review"],
            C_overflow=base_costs["C_overflow"],
            R_max=base_costs["R_max"],
            tau_binary=tau_binary,
            n_steps=60,
        )
        tau_lows.append(res.tau_low)
        tau_highs.append(res.tau_high)

    axes[0].plot(ratios, tau_lows,  "o-", color="#2ECC71", linewidth=1.8,
                 label="τ_low")
    axes[0].plot(ratios, tau_highs, "s-", color="#E74C3C", linewidth=1.8,
                 label="τ_high")
    axes[0].axhline(tau_binary, color="black", linestyle=":", linewidth=1.0,
                    label=f"τ_binary = {tau_binary:.3f}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("C_FN / C_FP ratio")
    axes[0].set_ylabel("Threshold value")
    axes[0].set_title("Threshold sensitivity to C_FN/C_FP ratio")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── Vary R_max ──
    r_maxes  = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    tau_lows2, tau_highs2 = [], []
    for r in r_maxes:
        res, _ = optimise_triclass_thresholds(
            y_true, y_prob,
            C_FN=base_costs["C_FN"], C_FP=base_costs["C_FP"],
            C_review=base_costs["C_review"],
            C_overflow=base_costs["C_overflow"],
            R_max=r,
            tau_binary=tau_binary,
            n_steps=60,
        )
        tau_lows2.append(res.tau_low)
        tau_highs2.append(res.tau_high)

    axes[1].plot([r * 100 for r in r_maxes], tau_lows2,  "o-", color="#2ECC71",
                 linewidth=1.8, label="τ_low")
    axes[1].plot([r * 100 for r in r_maxes], tau_highs2, "s-", color="#E74C3C",
                 linewidth=1.8, label="τ_high")
    axes[1].axhline(tau_binary, color="black", linestyle=":", linewidth=1.0,
                    label=f"τ_binary = {tau_binary:.3f}")
    axes[1].set_xlabel("R_max (%)")
    axes[1].set_ylabel("Threshold value")
    axes[1].set_title("Threshold sensitivity to review queue cap R_max")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "Sensitivity analysis — how stable are the thresholds under cost changes?",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 6 — 3-CLASS THRESHOLD OPTIMISATION")
    print("═" * 60)

    # 1. Load everything
    print("\n[1/7] Loading configs and data...")
    costs      = load_all_costs()
    tau_binary = load_binary_threshold()
    print(f"      τ_binary = {tau_binary:.4f}  (from Stage 5)")
    print(f"      C_FN={costs['C_FN']}  C_FP={costs['C_FP']}  "
          f"C_review={costs['C_review']}  C_overflow={costs['C_overflow']}  "
          f"R_max={costs['R_max']}")

    y_val  = np.load(str(ARTIFACTS / "y_val.npy"))
    y_prob = np.load(str(ARTIFACTS / "ensemble_val_probs.npy"))
    print(f"      Val set: {len(y_val):,}  fraud={y_val.sum()}")

    # 2. Grid search
    print(f"\n[2/7] Running joint grid search (100×100 = 10,000 evaluations)...")
    result, cost_grid = optimise_triclass_thresholds(
        y_true      = y_val,
        y_prob      = y_prob,
        C_FN        = costs["C_FN"],
        C_FP        = costs["C_FP"],
        C_review    = costs["C_review"],
        C_overflow  = costs["C_overflow"],
        R_max       = costs["R_max"],
        tau_binary  = tau_binary,
        n_steps     = 100,
    )

    print(f"\n      ┌──────────────────────────────────────────┐")
    print(f"      │  Optimal 3-class thresholds               │")
    print(f"      │  τ_low  = {result.tau_low:.4f}                       │")
    print(f"      │  τ_high = {result.tau_high:.4f}                       │")
    print(f"      ├──────────────────────────────────────────┤")
    print(f"      │  Composite cost  = {result.composite_cost:>12,.2f}          │")
    print(f"      │  Fraud escaped   = {result.fn_count:>5}  ({result.fn_rate*100:.2f}% of fraud) │")
    print(f"      │  Legit blocked   = {result.fp_count:>5}  ({result.fp_rate*100:.2f}% of legit) │")
    print(f"      │  Sent to review  = {result.review_count:>5}  ({result.review_rate*100:.2f}% of total) │")
    print(f"      └──────────────────────────────────────────┘")

    # 3. Write to config
    print(f"\n[3/7] Writing thresholds to decision config...")
    write_triclass_thresholds(result.tau_low, result.tau_high)

    # 4. Test set evaluation
    print(f"\n[4/7] Evaluating on TEST SET...")
    y_test     = np.load(str(ARTIFACTS / "y_test.npy"))
    test_probs = np.load(str(ARTIFACTS / "test_probs.npy"))

    n_test   = len(y_test)
    n_fraud  = int(y_test.sum())
    n_legit  = n_test - n_fraud

    auto_approve = test_probs < result.tau_low
    auto_block   = test_probs > result.tau_high
    review_mask  = (~auto_approve) & (~auto_block)

    test_fn         = int(((y_test==1) & auto_approve).sum())
    test_fp         = int(((y_test==0) & auto_block).sum())
    test_n_review   = int(review_mask.sum())
    test_review_rate= test_n_review / n_test

    print(f"\n      ── Test set 3-class results ──")
    print(f"      Fraud escaped (FN)  : {test_fn}  ({test_fn/n_fraud*100:.2f}% of fraud)")
    print(f"      Legit blocked (FP)  : {test_fp}  ({test_fp/n_legit*100:.2f}% of legit)")
    print(f"      Sent for review     : {test_n_review}  ({test_review_rate*100:.2f}% of total)")
    print(f"      Correctly blocked   : {int(((y_test==1)&auto_block).sum())}")
    print(f"      Correctly approved  : {int(((y_test==0)&auto_approve).sum())}")

    # 5. Sensitivity analysis
    print(f"\n[5/7] Running sensitivity analysis...")
    fig_sensitivity = plot_sensitivity_analysis(y_val, y_prob, costs, tau_binary)
    fig_sensitivity.savefig(PLOTS / "sensitivity_analysis.png", bbox_inches="tight", dpi=130)
    plt.close(fig_sensitivity)
    print(f"      Saved → outputs/plots/sensitivity_analysis.png")

    # 6. Plots
    print(f"\n[6/7] Saving plots...")

    # Reconstruct grids for heatmap labels
    low_grid  = np.linspace(0.01, max(tau_binary - 0.01, 0.02), 100)
    high_grid = np.linspace(min(tau_binary + 0.01, 0.98), 0.99, 100)

    fig_grid = plot_cost_grid(cost_grid, low_grid, high_grid,
                               result.tau_low, result.tau_high, tau_binary)
    fig_grid.savefig(PLOTS / "cost_grid_triclass.png", bbox_inches="tight", dpi=130)
    plt.close(fig_grid)

    fig_analysis = plot_triclass_analysis(y_val, y_prob,
                                           result.tau_low, result.tau_high,
                                           tau_binary, result)
    fig_analysis.savefig(PLOTS / "triclass_analysis.png", bbox_inches="tight", dpi=130)
    plt.close(fig_analysis)
    print(f"      Saved → outputs/plots/cost_grid_triclass.png")
    print(f"      Saved → outputs/plots/triclass_analysis.png")

    # 7. MLflow
    print(f"\n[7/7] Logging to MLflow...")
    setup_mlflow()
    end_run_if_active()
    with mlflow.start_run(run_name="stage6_triclass_threshold"):
        mlflow.log_params({
            "tau_binary":  tau_binary,
            "tau_low":     result.tau_low,
            "tau_high":    result.tau_high,
            **{f"cost_{k}": v for k, v in costs.items()},
        })
        mlflow.log_metrics({
            "val_composite_cost": result.composite_cost,
            "val_fn_rate":        result.fn_rate,
            "val_fp_rate":        result.fp_rate,
            "val_review_rate":    result.review_rate,
            "test_fn_count":      test_fn,
            "test_fp_count":      test_fp,
            "test_review_count":  test_n_review,
            "test_review_rate":   test_review_rate,
        })
        for fname in ["cost_grid_triclass.png", "triclass_analysis.png",
                      "sensitivity_analysis.png"]:
            mlflow.log_artifact(str(PLOTS / fname), "plots")
        mlflow.log_artifact(str(CONFIG), "configs")

    print("\n" + "─" * 60)
    print("  Stage 6 complete.")
    print(f"  τ_low  = {result.tau_low:.4f}")
    print(f"  τ_high = {result.tau_high:.4f}")
    print(f"  Both written to configs/decision_config.yaml")
    print("  Next step: python stage7_api.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()