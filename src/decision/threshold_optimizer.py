"""
src/decision/threshold_optimizer.py
─────────────────────────────────────
Cost-aware threshold optimisation for binary and 3-class decision rules.

Binary (Stage 5):
─────────────────
    p >= τ  → Fraud
    p <  τ  → Legitimate

    ExpectedCost(τ) = C_FN × FN(τ) + C_FP × FP(τ)

    We sweep τ ∈ [0.01, 0.99] and pick the τ that minimises expected cost
    on the validation set. The result is stored in decision_config.yaml.

3-Class (Stage 6):
──────────────────
    p > τ_high        → Fraud (auto-block)
    τ_low ≤ p ≤ τ_high → Human review
    p < τ_low         → Legitimate (auto-approve)

    CompositeCost(τ_low, τ_high) =
        C_FN     × count(fraud with p < τ_low)            # escaped fraud
      + C_FP     × count(legit with p > τ_high)           # wrongly blocked
      + C_review × count(τ_low ≤ p ≤ τ_high)             # reviewer workload
      + C_overflow × max(0, review_rate - R_max)          # queue penalty

    Constraints enforced:
      τ_low < τ_high
      τ_low ≥ 0.01, τ_high ≤ 0.99

    Search: joint grid search over τ_low × τ_high, Δ=0.01 resolution.
    The binary τ* from Stage 5 is used to constrain the grid:
      τ_low  ∈ [0.01, τ* - 0.01]
      τ_high ∈ [τ* + 0.01, 0.99]
    This enforces the ordering τ_low < τ* < τ_high so the human review
    band always straddles the binary decision boundary.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


# ─── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class BinaryThresholdResult:
    tau:              float
    expected_cost:    float
    fn_count:         int
    fp_count:         int
    fn_rate:          float   # FN / n_actual_fraud
    fp_rate:          float   # FP / n_actual_legit
    precision:        float
    recall:           float
    f1:               float


@dataclass
class TriClassThresholdResult:
    tau_low:          float
    tau_high:         float
    composite_cost:   float
    fn_count:         int     # fraud auto-approved (escaped)
    fp_count:         int     # legit auto-blocked
    review_count:     int     # sent to human review
    review_rate:      float   # review_count / n_total
    fn_rate:          float
    fp_rate:          float


# ─── Binary threshold optimisation ───────────────────────────────────────────

def optimise_binary_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    C_FN: float,
    C_FP: float,
    n_steps: int = 200,
) -> tuple[BinaryThresholdResult, np.ndarray, np.ndarray]:
    """
    Sweeps τ and returns the threshold minimising expected cost.

    Args:
        y_true:  True binary labels.
        y_prob:  Ensemble calibrated probabilities.
        C_FN:    Cost of a false negative (missed fraud).
        C_FP:    Cost of a false positive (wrongly blocked legit txn).
        n_steps: Number of threshold values to sweep.

    Returns:
        best_result:  BinaryThresholdResult at optimal τ.
        taus:         Array of all swept threshold values.
        costs:        Array of expected costs at each τ.
    """
    assert C_FN is not None and C_FP is not None, (
        "C_FN and C_FP must be set in configs/cost_config.yaml before running "
        "threshold optimisation."
    )

    taus  = np.linspace(0.01, 0.99, n_steps)
    costs = np.zeros(n_steps)

    n_fraud = int(y_true.sum())
    n_legit = int((y_true == 0).sum())

    for i, tau in enumerate(taus):
        y_pred = (y_prob >= tau).astype(int)
        fn     = int(((y_true == 1) & (y_pred == 0)).sum())
        fp     = int(((y_true == 0) & (y_pred == 1)).sum())
        costs[i] = C_FN * fn + C_FP * fp

    best_i   = int(np.argmin(costs))
    best_tau = float(taus[best_i])
    y_pred   = (y_prob >= best_tau).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)

    result = BinaryThresholdResult(
        tau           = best_tau,
        expected_cost = float(costs[best_i]),
        fn_count      = fn,
        fp_count      = fp,
        fn_rate       = fn / max(n_fraud, 1),
        fp_rate       = fp / max(n_legit, 1),
        precision     = prec,
        recall        = rec,
        f1            = f1,
    )
    return result, taus, costs


# ─── 3-class threshold optimisation ─────────────────────────────────────────

def optimise_triclass_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    C_FN: float,
    C_FP: float,
    C_review: float,
    C_overflow: float,
    R_max: float,
    tau_binary: float,
    n_steps: int = 100,
) -> tuple[TriClassThresholdResult, np.ndarray]:
    """
    Joint grid search over (τ_low, τ_high) minimising the composite cost.

    The binary τ* anchors the search:
        τ_low  searches in [0.01, τ* - 0.01]
        τ_high searches in [τ* + 0.01, 0.99]

    Args:
        y_true:      True binary labels.
        y_prob:      Ensemble calibrated probabilities.
        C_FN:        Cost of missed fraud.
        C_FP:        Cost of wrongly blocked legit.
        C_review:    Cost of one human review.
        C_overflow:  Penalty per unit review_rate above R_max.
        R_max:       Max tolerable review rate (e.g. 0.03 = 3%).
        tau_binary:  Binary threshold from Stage 5 — anchors the search space.
        n_steps:     Grid resolution per axis (n_steps² evaluations total).

    Returns:
        best_result:  TriClassThresholdResult at optimal (τ_low, τ_high).
        cost_grid:    2-D array of composite costs, shape (n_low, n_high).
    """
    n_total = len(y_true)

    # Search grids anchored around the binary threshold
    low_grid  = np.linspace(0.01, max(tau_binary - 0.01, 0.02), n_steps)
    high_grid = np.linspace(min(tau_binary + 0.01, 0.98), 0.99, n_steps)

    cost_grid  = np.full((len(low_grid), len(high_grid)), np.inf)
    best_cost  = np.inf
    best_i, best_j = 0, 0

    for i, tau_low in enumerate(low_grid):
        for j, tau_high in enumerate(high_grid):
            if tau_low >= tau_high:
                continue

            # Classify each transaction
            auto_approve = y_prob < tau_low
            auto_block   = y_prob > tau_high
            review_mask  = (~auto_approve) & (~auto_block)

            fn          = int(((y_true == 1) & auto_approve).sum())   # fraud escaped
            fp          = int(((y_true == 0) & auto_block).sum())     # legit blocked
            n_review    = int(review_mask.sum())
            review_rate = n_review / max(n_total, 1)

            cost = (
                C_FN     * fn
              + C_FP     * fp
              + C_review * n_review
              + C_overflow * max(0.0, review_rate - R_max)
            )

            cost_grid[i, j] = cost

            if cost < best_cost:
                best_cost  = cost
                best_i, best_j = i, j

    best_tau_low  = float(low_grid[best_i])
    best_tau_high = float(high_grid[best_j])

    # Recompute stats at best thresholds
    auto_approve = y_prob < best_tau_low
    auto_block   = y_prob > best_tau_high
    review_mask  = (~auto_approve) & (~auto_block)

    fn          = int(((y_true == 1) & auto_approve).sum())
    fp          = int(((y_true == 0) & auto_block).sum())
    n_review    = int(review_mask.sum())
    review_rate = n_review / max(n_total, 1)
    n_fraud     = int(y_true.sum())
    n_legit     = int((y_true == 0).sum())

    result = TriClassThresholdResult(
        tau_low       = best_tau_low,
        tau_high      = best_tau_high,
        composite_cost= best_cost,
        fn_count      = fn,
        fp_count      = fp,
        review_count  = n_review,
        review_rate   = review_rate,
        fn_rate       = fn / max(n_fraud, 1),
        fp_rate       = fp / max(n_legit, 1),
    )
    return result, cost_grid


# ─── Rolling review rate (runtime monitoring) ─────────────────────────────────

def rolling_review_rate(
    decisions: list[int],
    window: int = 100,
) -> float:
    """
    Fraction of the last `window` decisions that were Class 1 (human review).
    Used as a runtime monitoring metric by the API.

    Args:
        decisions: List of integer decisions (0=approve, 1=review, 2=block).
        window:    Rolling window size.

    Returns:
        Review rate as a float in [0, 1].
    """
    recent = decisions[-window:] if len(decisions) >= window else decisions
    if not recent:
        return 0.0
    return sum(1 for d in recent if d == 1) / len(recent)


# ─── Decision function (used at inference time) ───────────────────────────────

def classify_transaction(
    p: float,
    tau_low: float,
    tau_high: float,
) -> tuple[int, str]:
    """
    Applies the 3-class decision rule to a single probability.

    Returns:
        (class_int, class_label)
        0 → "legitimate"
        1 → "human_review"
        2 → "fraud"
    """
    if p > tau_high:
        return 2, "fraud"
    elif p < tau_low:
        return 0, "legitimate"
    else:
        return 1, "human_review"
