"""
src/monitoring/drift.py
────────────────────────
Drift detection using Population Stability Index (PSI).

PSI measures how much a distribution has shifted between a reference window
(training data) and a current window (recent production data).

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

Interpretation:
    PSI < 0.1   → No significant drift, model stable
    PSI 0.1–0.2 → Moderate drift, monitor closely
    PSI > 0.2   → Significant drift, trigger retraining

We compute PSI on two things:
    1. Input features (V1–V28, Amount, Time) — detects covariate shift.
       If the distribution of incoming transactions changes (e.g. new
       spending patterns, seasonal effects), feature PSI catches it first.

    2. Score distribution (ensemble output probability) — detects concept
       drift. Even if features look stable, if the model's output
       distribution shifts it means the relationship between features and
       fraud has changed.

Score PSI is the most actionable signal: a jump in score PSI means the model
is encountering transactions it was not calibrated for, regardless of why.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ─── PSI computation ──────────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Computes PSI between a reference distribution and a current distribution.

    Bins are derived from the reference distribution (equal-frequency binning
    so that every bin has roughly the same number of reference samples —
    this avoids sparse bins at the tails that inflate PSI artificially).

    Args:
        reference: 1-D array from the reference period (training distribution).
        current:   1-D array from the current period (production window).
        n_bins:    Number of quantile bins.
        eps:       Small constant to avoid log(0).

    Returns:
        PSI value (float).
    """
    # Build bin edges from reference quantiles
    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.percentile(reference, quantiles)
    bin_edges[0]  -= 1e-8    # include the minimum
    bin_edges[-1] += 1e-8    # include the maximum

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Clip to avoid log(0)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_feature_psi(
    X_reference: np.ndarray,
    X_current: np.ndarray,
    feature_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Computes PSI for every feature column independently.

    time_sin and time_cos are excluded: they encode time-of-day from the raw
    Time column. Because train/test cover different calendar windows, their
    time-of-day distributions naturally differ — this is not model-relevant
    drift and produces misleadingly large PSI values (10+).
    Only V1–V28 and amount_scaled are monitored.

    Args:
        X_reference:   (n_ref, n_features) reference feature matrix.
        X_current:     (n_cur, n_features) current feature matrix.
        feature_names: Optional list of feature names for the output dict keys.
        n_bins:        Number of quantile bins per feature.

    Returns:
        { feature_name: psi_value }  (time_sin and time_cos excluded)
    """
    n_features = X_reference.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Exclude cyclical time features — they drift trivially across time periods
    excluded = {"time_sin", "time_cos"}

    return {
        name: compute_psi(X_reference[:, i], X_current[:, i], n_bins=n_bins)
        for i, name in enumerate(feature_names)
        if name not in excluded
    }


# ─── Drift result ─────────────────────────────────────────────────────────────

@dataclass
class DriftReport:
    """
    Full drift report for one monitoring window.

    Attributes:
        score_psi:       PSI on the model's output probability distribution.
        feature_psi:     Per-feature PSI values.
        max_feature_psi: Highest feature PSI (most drifted feature).
        max_feature_name:Name of the most drifted feature.
        score_status:    "stable" / "warning" / "drift"
        feature_status:  "stable" / "warning" / "drift"  (based on max feature PSI)
        window_size:     Number of transactions in the current window.
        retraining_recommended: True if score PSI > 0.2.
    """
    score_psi:              float
    feature_psi:            dict[str, float]
    max_feature_psi:        float
    max_feature_name:       str
    score_status:           str
    feature_status:         str
    window_size:            int
    retraining_recommended: bool
    timestamp:              str = ""

    def summary(self) -> str:
        lines = [
            f"── Drift Report ({'⚠ RETRAIN' if self.retraining_recommended else 'OK'}) ──",
            f"  Score PSI    : {self.score_psi:.4f}  [{self.score_status}]",
            f"  Max feat PSI : {self.max_feature_psi:.4f}  [{self.feature_status}]"
            f"  ({self.max_feature_name})",
            f"  Window size  : {self.window_size:,}",
        ]
        if self.retraining_recommended:
            lines.append("  !! Retraining recommended: score PSI > 0.2 !!")
        return "\n".join(lines)


def _psi_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    elif psi < 0.2:
        return "warning"
    else:
        return "drift"


def build_drift_report(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    reference_features: np.ndarray,
    current_features: np.ndarray,
    feature_names: list[str] | None = None,
    timestamp: str = "",
) -> DriftReport:
    """
    Builds a full DriftReport comparing reference and current windows.
    """
    score_psi   = compute_psi(reference_scores, current_scores)
    feature_psi = compute_feature_psi(
        reference_features, current_features, feature_names=feature_names
    )

    max_feat_name = max(feature_psi, key=feature_psi.get)
    max_feat_psi  = feature_psi[max_feat_name]

    return DriftReport(
        score_psi              = score_psi,
        feature_psi            = feature_psi,
        max_feature_psi        = max_feat_psi,
        max_feature_name       = max_feat_name,
        score_status           = _psi_status(score_psi),
        feature_status         = _psi_status(max_feat_psi),
        window_size            = len(current_scores),
        retraining_recommended = score_psi > 0.2,
        timestamp              = timestamp,
    )