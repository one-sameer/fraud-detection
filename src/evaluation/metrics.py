"""
src/evaluation/metrics.py
─────────────────────────
All evaluation functions used across every stage of the project.

Primary metric: AUPRC (Area Under Precision-Recall Curve).
We use AUPRC — not AUROC — as the headline metric because AUROC is
optimistic under extreme class imbalance. A model that predicts "not fraud"
for every transaction scores ~0.998 AUROC on this dataset. AUPRC correctly
penalises a model that never finds fraud.

Secondary metrics:
  - MCC (Matthews Correlation Coefficient): single scalar balanced for
    imbalanced classes. 1.0 = perfect, 0.0 = random, -1.0 = inverse.
  - ECE (Expected Calibration Error): measures how well raw probabilities
    reflect true empirical frequencies.
  - F1, Precision, Recall at a given threshold.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve


# ─── Core metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Full metric suite for a model outputting probabilities.

    Args:
        y_true:    Ground-truth binary labels (0/1).
        y_prob:    Predicted fraud probabilities in [0, 1].
        threshold: Decision boundary for label-dependent metrics.

    Returns:
        Dictionary of all metrics. Threshold-independent metrics (AUPRC,
        AUROC) are always computed. Threshold-dependent metrics (F1, MCC,
        precision, recall, confusion matrix entries) use the given threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)

    auprc = average_precision_score(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    mcc   = matthews_corrcoef(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auprc":     round(auprc, 6),
        "auroc":     round(auroc, 6),
        "mcc":       round(mcc, 6),
        "f1":        round(f1, 6),
        "precision": round(prec, 6),
        "recall":    round(rec, 6),
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "n_predicted_fraud": int(tp + fp),
        "n_actual_fraud":    int(tp + fn),
    }


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Sweeps all PR-curve thresholds and returns (best_threshold, best_f1).

    Useful for initial model evaluation. The cost-aware threshold search
    in stage5 supersedes this for production threshold selection.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    denom = precisions + recalls
    f1_scores = np.where(denom > 0, 2 * precisions * recalls / denom, 0.0)
    best_idx = int(np.argmax(f1_scores[:-1]))  # last entry has no threshold
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ─── Calibration metrics ──────────────────────────────────────────────────────

def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Expected Calibration Error.

    Partitions [0, 1] into n_bins equal-width bins. For each bin, measures
    the absolute gap between the mean predicted probability and the observed
    fraud rate. The ECE is the weighted average of these gaps (weighted by
    bin population).

    A perfectly calibrated model scores ECE = 0.
    On this dataset, fraud is rare so bins near 0 are densely populated
    and bins near 1 are sparse — cross-validated calibration addresses this.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        mean_conf = float(y_prob[mask].mean())
        empirical = float(y_true[mask].mean())
        ece += (mask.sum() / n) * abs(mean_conf - empirical)

    return round(float(ece), 6)


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    n_bins: int = 20,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Reliability (calibration) diagram.

    A well-calibrated model's curve lies on the y=x diagonal.
    Curves above the diagonal → model is under-confident.
    Curves below the diagonal → model is over-confident.
    """
    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.get_figure()

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration", alpha=0.6)
    ax.plot(
        prob_pred, prob_true,
        "o-", linewidth=1.5, markersize=4,
        label=f"{model_name}  (ECE = {ece:.4f})"
    )
    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Fraction of positives (observed)", fontsize=10)
    ax.set_title(f"Reliability diagram — {model_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig


def plot_pr_curve(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall curves",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plots PR curves for multiple models on the same axes.

    Args:
        results: { model_name: (y_true, y_prob) }
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    for name, (y_true, y_prob) in results.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, linewidth=1.5, label=f"{name}  (AUPRC = {auprc:.4f})")

    # Baseline: random classifier at fraud prevalence
    prevalence = float(y_true.mean())
    ax.axhline(prevalence, linestyle=":", color="gray", linewidth=0.8,
               label=f"Random baseline ({prevalence:.3f})")

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    return fig


def print_metrics_table(metrics_dict: dict[str, dict]) -> None:
    """Pretty-prints a comparison table of metrics across models."""
    header = f"{'Model':<25} {'AUPRC':>8} {'AUROC':>8} {'MCC':>8} {'F1':>8} {'Recall':>8} {'ECE':>8}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for name, m in metrics_dict.items():
        ece_str = f"{m.get('ece', float('nan')):.6f}"
        print(
            f"{name:<25} "
            f"{m['auprc']:>8.6f} "
            f"{m['auroc']:>8.6f} "
            f"{m['mcc']:>8.6f} "
            f"{m['f1']:>8.6f} "
            f"{m['recall']:>8.6f} "
            f"{ece_str:>8}"
        )
    print("─" * len(header) + "\n")
