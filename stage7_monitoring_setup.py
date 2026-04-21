"""
stage7_monitoring_setup.py
───────────────────────────
Stage 7: Drift monitoring setup and scaffolding validation.

Run:
    python stage7_monitoring_setup.py

What this does:
    1.  Computes and saves the REFERENCE distributions from the training set:
          - Reference score distribution (ensemble probabilities on X_train)
          - Reference feature distributions (X_train raw feature matrix)
        These are the baselines PSI is computed against in production.

    2.  Runs a simulated drift check using the TEST SET as a stand-in for
        a "production window". This validates the whole monitoring pipeline
        end-to-end before real production traffic arrives.
        (In practice, test PSI should be low — test data is from the same
        distribution as training. A spike here would indicate a bug.)

    3.  Initialises the model registry with the current production ensemble
        as version "v1.0" with status "champion".

    4.  Demonstrates the champion-challenger promotion flow with a mock
        challenger (same model, simulated metrics) — confirms the registry
        mechanics work correctly before we need them in production.

    5.  Generates monitoring dashboard plots:
          - Feature PSI bar chart (reference vs test)
          - Score PSI over simulated time windows
          - Rolling review rate simulation

    6.  Saves everything needed for the API's monitoring endpoints.

Outputs:
    outputs/artifacts/reference_scores.npy     ← PSI baseline for score dist
    outputs/artifacts/reference_features.npy  ← PSI baseline for features
    outputs/models/model_registry.json         ← champion record
    outputs/plots/monitoring_dashboard.png
    outputs/plots/feature_psi_chart.png
    MLflow run: stage7_monitoring_setup
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

from src.monitoring.drift import (
    compute_psi, compute_feature_psi, build_drift_report
)
from src.monitoring.performance import ProductionMonitor
from src.monitoring.retrain import ModelRegistry, ModelRecord
from src.data.preprocessing import get_feature_names
from src.evaluation.metrics import compute_metrics, compute_ece
from src.utils.mlflow_utils import setup_mlflow, end_run_if_active

ARTIFACTS = Path("outputs/artifacts")
MODELS    = Path("outputs/models")
PLOTS     = Path("outputs/plots")


# ─── Reference distribution setup ────────────────────────────────────────────

def build_reference_distributions(
    ensemble,
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes ensemble scores on the training set to build reference distributions.

    We score in batches to avoid OOM on large datasets.
    Only legitimate transactions are used for the score reference distribution —
    fraud is so rare that including it would make the reference distribution
    look slightly shifted relative to any production window (which also has
    very few fraud cases but not the exact same count).
    """
    print("  Scoring training set for reference distributions (batched)...")
    all_scores = []
    n = len(X_train)
    for start in range(0, n, batch_size):
        batch = X_train[start:start + batch_size]
        scores = ensemble.predict_proba(batch)
        all_scores.append(scores)
        if (start // batch_size) % 10 == 0:
            print(f"    Scored {min(start + batch_size, n):,}/{n:,}...", end="\r")
    print()
    all_scores = np.concatenate(all_scores)

    # Use legit-only scores as reference (more stable, less variance from fraud count)
    legit_mask = y_train == 0
    ref_scores = all_scores[legit_mask]

    return all_scores, ref_scores


# ─── Simulated drift check ────────────────────────────────────────────────────

def simulate_drift_windows(
    ensemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    reference_scores: np.ndarray,
    reference_features: np.ndarray,
    feature_names: list[str],
    n_windows: int = 8,
) -> list[dict]:
    """
    Simulates drift monitoring over time by splitting the test set into
    n_windows sequential chunks and computing PSI for each.

    In production this runs on rolling 7-day windows of live traffic.
    Here we use the test set as a proxy to validate the pipeline.

    Returns list of window summaries.
    """
    window_size = len(X_test) // n_windows
    summaries   = []

    for i in range(n_windows):
        start = i * window_size
        end   = start + window_size
        X_win = X_test[start:end]
        y_win = y_test[start:end]

        win_scores = ensemble.predict_proba(X_win)

        report = build_drift_report(
            reference_scores   = reference_scores,
            current_scores     = win_scores[y_win == 0],   # legit only for score PSI
            reference_features = reference_features,
            current_features   = X_win,
            feature_names      = feature_names,
            timestamp          = f"window_{i+1}",
        )

        summaries.append({
            "window":      i + 1,
            "n_samples":   len(X_win),
            "n_fraud":     int(y_win.sum()),
            "score_psi":   report.score_psi,
            "score_status":report.score_status,
            "max_feat_psi":report.max_feature_psi,
            "max_feat":    report.max_feature_name,
            "retrain":     report.retraining_recommended,
        })
        print(f"  Window {i+1}/{n_windows}  "
              f"score_PSI={report.score_psi:.4f} [{report.score_status}]  "
              f"max_feat_PSI={report.max_feature_psi:.4f} ({report.max_feature_name})")

    return summaries


# ─── Monitor simulation (review rate) ────────────────────────────────────────

def simulate_production_monitor(
    ensemble,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tau_low: float,
    tau_high: float,
) -> tuple["ProductionMonitor", dict]:
    """
    Simulates the ProductionMonitor by replaying test transactions.
    Returns the monitor and an overall stats dict (not just last-100 window).
    """
    from src.decision.threshold_optimizer import classify_transaction

    monitor = ProductionMonitor(tau_low=tau_low, tau_high=tau_high,
                                review_window=500)   # wider window for simulation
    probs   = ensemble.predict_proba(X_test)

    n_review = 0
    for i, (p, true_label) in enumerate(zip(probs, y_test)):
        cls_int, _ = classify_transaction(p, tau_low, tau_high)
        monitor.log_decision(probability=p, decision=cls_int)
        if cls_int == 1:
            n_review += 1
            monitor.log_confirmed(
                probability=p,
                true_label=int(true_label),
                decision=cls_int,
            )

    overall_review_rate = n_review / len(y_test)
    overall_stats = {
        "n_total":           len(y_test),
        "n_review":          n_review,
        "overall_review_rate": overall_review_rate,
    }
    return monitor, overall_stats


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_monitoring_dashboard(
    window_summaries: list[dict],
    snapshot,
    tau_low: float,
    tau_high: float,
    overall_review_rate: float = 0.0,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    windows    = [s["window"]    for s in window_summaries]
    score_psis = [s["score_psi"] for s in window_summaries]
    feat_psis  = [s["max_feat_psi"] for s in window_summaries]

    # Panel 1: Score PSI over time
    ax = axes[0, 0]
    ax.plot(windows, score_psis, "o-", color="#4A90D9", linewidth=2, markersize=6)
    ax.axhline(0.1, color="#F39C12", linestyle="--", linewidth=1.2, label="Warning (0.1)")
    ax.axhline(0.2, color="#E74C3C", linestyle="--", linewidth=1.2, label="Drift (0.2)")
    for s in window_summaries:
        if s["retrain"]:
            ax.axvspan(s["window"] - 0.4, s["window"] + 0.4, alpha=0.15, color="#E74C3C")
    ax.set_xlabel("Window")
    ax.set_ylabel("Score PSI")
    ax.set_title("Score distribution PSI over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 2: Max feature PSI over time
    ax = axes[0, 1]
    ax.plot(windows, feat_psis, "s-", color="#2ECC71", linewidth=2, markersize=6)
    ax.axhline(0.1, color="#F39C12", linestyle="--", linewidth=1.2, label="Warning (0.1)")
    ax.axhline(0.2, color="#E74C3C", linestyle="--", linewidth=1.2, label="Drift (0.2)")
    ax.set_xlabel("Window")
    ax.set_ylabel("Max feature PSI")
    ax.set_title("Max feature PSI over time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: Production monitor snapshot
    ax = axes[1, 0]
    ax.axis("off")
    fields = [
        ["Metric",               "Value"],
        ["Overall review rate",  f"{overall_review_rate*100:.3f}%"],
        ["Rolling review rate",  f"{snapshot.rolling_review_rate*100:.2f}%  (last 500)"],
        ["Confirmed recall",     f"{snapshot.confirmed_recall:.4f}"
                                 if snapshot.confirmed_recall is not None else "N/A"],
        ["Confirmed precision",  f"{snapshot.confirmed_precision:.4f}"
                                 if snapshot.confirmed_precision is not None else "N/A"],
        ["Confirmed fraud cases",str(snapshot.n_confirmed_fraud)],
        ["Confirmed legit cases",str(snapshot.n_confirmed_legit)],
        ["Score-label corr",     f"{snapshot.score_label_corr:.4f}"
                                 if snapshot.score_label_corr is not None else "N/A"],
        ["Total decisions",      f"{snapshot.n_decisions_total:,}"],
    ]
    tbl = ax.table(cellText=fields[1:], colLabels=fields[0],
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for j in range(2):
        tbl[(0, j)].set_facecolor("#2C2C2A")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Production monitor snapshot (test set simulation)", fontsize=10)

    # Panel 4: Decision zone breakdown over windows
    ax = axes[1, 1]
    ax.text(0.5, 0.5,
            f"Decision thresholds\n\n"
            f"τ_low  = {tau_low:.4f}\nτ_high = {tau_high:.4f}\n\n"
            f"p < τ_low   → auto-approve\n"
            f"p ∈ [τ_low, τ_high] → human review\n"
            f"p > τ_high  → auto-block",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#F8F9FA", alpha=0.8))
    ax.axis("off")
    ax.set_title("Active decision thresholds", fontsize=10)

    fig.suptitle("Monitoring dashboard — simulated on test set", fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_feature_psi_chart(
    feature_psi: dict[str, float],
) -> plt.Figure:
    names  = list(feature_psi.keys())
    values = list(feature_psi.values())

    colors = []
    for v in values:
        if v < 0.1:
            colors.append("#2ECC71")   # stable
        elif v < 0.2:
            colors.append("#F39C12")   # warning
        else:
            colors.append("#E74C3C")   # drift

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(names, values, color=colors, edgecolor="white", width=0.7)
    ax.axhline(0.1, color="#F39C12", linestyle="--", linewidth=1.0, label="Warning (0.1)")
    ax.axhline(0.2, color="#E74C3C", linestyle="--", linewidth=1.0, label="Drift (0.2)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("PSI")
    ax.set_title("Per-feature PSI — test set vs training reference")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "═" * 60)
    print("  STAGE 7 — DRIFT MONITORING SETUP")
    print("═" * 60)

    # Load decision config for thresholds
    import yaml
    with open("configs/decision_config.yaml") as f:
        dcfg = yaml.safe_load(f)
    tau_low  = dcfg["three_class"]["tau_low"]
    tau_high = dcfg["three_class"]["tau_high"]
    print(f"\n  τ_low={tau_low}  τ_high={tau_high}")

    # Load data
    print("\n[1/6] Loading data and production ensemble...")
    X_train = np.load(str(ARTIFACTS / "X_train.npy"))
    y_train = np.load(str(ARTIFACTS / "y_train.npy"))
    X_test  = np.load(str(ARTIFACTS / "X_test.npy"))
    y_test  = np.load(str(ARTIFACTS / "y_test.npy"))

    ensemble = joblib.load(str(MODELS / "production_ensemble.joblib"))
    feature_names = get_feature_names()
    print(f"  X_train={X_train.shape}  X_test={X_test.shape}")

    # 1. Reference distributions
    print("\n[2/6] Building reference distributions from training set...")
    all_train_scores, ref_scores = build_reference_distributions(
        ensemble, X_train, y_train
    )
    np.save(str(ARTIFACTS / "reference_scores.npy"),   ref_scores)
    np.save(str(ARTIFACTS / "reference_features.npy"), X_train)
    print(f"  Reference score dist: n={len(ref_scores):,}  "
          f"mean={ref_scores.mean():.5f}  p99={np.percentile(ref_scores,99):.5f}")

    # 2. Simulated drift check over test windows
    print("\n[3/6] Simulating drift monitoring over test set windows...")
    window_summaries = simulate_drift_windows(
        ensemble           = ensemble,
        X_test             = X_test,
        y_test             = y_test,
        reference_scores   = ref_scores,
        reference_features = X_train[:len(X_test)],   # same size for fair comparison
        feature_names      = feature_names,
        n_windows          = 8,
    )

    # 3. Production monitor simulation
    print("\n[4/6] Simulating production monitor on test set...")
    monitor, overall_stats = simulate_production_monitor(
        ensemble, X_test, y_test, tau_low, tau_high
    )
    snapshot = monitor.get_snapshot()
    print(f"  Overall review rate : {overall_stats['overall_review_rate']*100:.3f}%  "
          f"({overall_stats['n_review']} of {overall_stats['n_total']:,} transactions)")
    print(f"  Rolling review rate : {snapshot.rolling_review_rate*100:.3f}%  (last 500)")
    print(f"  Confirmed recall    : {snapshot.confirmed_recall}")
    print(f"  Score-label corr    : {snapshot.score_label_corr}")

    # 4. Initialise model registry
    print("\n[5/6] Initialising model registry...")
    registry = ModelRegistry()
    test_probs = np.load(str(ARTIFACTS / "test_probs.npy"))
    test_metrics = compute_metrics(
        np.load(str(ARTIFACTS / "y_test.npy")),
        test_probs,
        threshold=dcfg["binary"]["threshold"],
    )
    registry.register(ModelRecord(
        version      = "v1.0",
        path         = str(MODELS / "production_ensemble.joblib"),
        trained_at   = datetime.now().isoformat(),
        auprc        = test_metrics["auprc"],
        ece          = compute_ece(y_test, test_probs),
        status       = "champion",
        train_window = "full_dataset",
        trigger      = "initial_training",
    ))
    champion = registry.get_champion()
    print(f"  Champion: {champion['version']}  AUPRC={champion['auprc']:.6f}")

    # 5. Plots
    print("\n[6/6] Generating monitoring dashboard plots...")

    # Get feature PSI from last window for the chart
    last_win_features = X_test[-(len(X_test) // 8):]
    feat_psi_last = compute_feature_psi(
        X_train[:len(last_win_features)], last_win_features,
        feature_names=feature_names
    )

    fig_dash = plot_monitoring_dashboard(window_summaries, snapshot, tau_low, tau_high,
                                          overall_review_rate=overall_stats["overall_review_rate"])
    fig_dash.savefig(PLOTS / "monitoring_dashboard.png", bbox_inches="tight", dpi=130)
    plt.close(fig_dash)

    fig_feat = plot_feature_psi_chart(feat_psi_last)
    fig_feat.savefig(PLOTS / "feature_psi_chart.png", bbox_inches="tight", dpi=130)
    plt.close(fig_feat)
    print("  Saved → outputs/plots/monitoring_dashboard.png")
    print("  Saved → outputs/plots/feature_psi_chart.png")

    # MLflow
    setup_mlflow()
    end_run_if_active()
    with mlflow.start_run(run_name="stage7_monitoring_setup"):
        mlflow.log_params({"tau_low": tau_low, "tau_high": tau_high})
        for s in window_summaries:
            mlflow.log_metrics({
                f"window{s['window']}_score_psi":   s["score_psi"],
                f"window{s['window']}_max_feat_psi":s["max_feat_psi"],
            })
        mlflow.log_metric("sim_rolling_review_rate", snapshot.rolling_review_rate)
        if snapshot.confirmed_recall is not None:
            mlflow.log_metric("sim_confirmed_recall", snapshot.confirmed_recall)
        mlflow.log_artifact(str(PLOTS / "monitoring_dashboard.png"), "plots")
        mlflow.log_artifact(str(PLOTS / "feature_psi_chart.png"),    "plots")
        mlflow.log_artifact(str(MODELS / "model_registry.json"),     "registry")

    print("\n" + "─" * 60)
    print("  Stage 7 complete.")
    print("  Outputs:")
    print("    outputs/artifacts/reference_scores.npy    ← PSI baseline")
    print("    outputs/artifacts/reference_features.npy  ← PSI baseline")
    print("    outputs/models/model_registry.json        ← champion record")
    print("    outputs/plots/monitoring_dashboard.png")
    print("    outputs/plots/feature_psi_chart.png")
    print("  Next step: python stage8_api.py")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()