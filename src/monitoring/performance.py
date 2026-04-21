"""
src/monitoring/performance.py
──────────────────────────────
Production performance monitoring using confirmed fraud labels.

In production, some transactions that go to human review get confirmed as
fraud or cleared as legitimate. These confirmed labels are the ground truth
signal that lets us track whether the model's actual recall is degrading.

This module maintains a rolling buffer of (probability, true_label) pairs
from confirmed cases, and computes:

    1. Confirmed recall  — of fraud cases that got a confirmed label,
                           what fraction did the model score above τ_low?
                           (i.e. did not auto-approve as legitimate)
    2. Rolling review rate — fraction of the last N decisions classified
                             as human review. Spike = thresholds need tuning.
    3. Score-label correlation — Pearson r between model score and confirmed
                                 label. Falling r signals concept drift.

The monitor does NOT make retraining decisions — it exposes metrics that
the drift detection layer (drift.py) and the retraining orchestrator
(retrain.py) consume.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field


@dataclass
class ConfirmedCase:
    """One transaction with a confirmed label from human review."""
    probability:  float
    true_label:   int     # 0 = legit, 1 = fraud
    model_decision: int   # 0 = approve, 1 = review, 2 = block
    timestamp:    str = ""


@dataclass
class PerformanceSnapshot:
    confirmed_recall:      float | None   # None if no confirmed fraud cases yet
    confirmed_precision:   float | None
    n_confirmed_fraud:     int
    n_confirmed_legit:     int
    rolling_review_rate:   float
    score_label_corr:      float | None
    n_decisions_total:     int


class ProductionMonitor:
    """
    Maintains rolling buffers of production decisions and confirmed labels.

    Usage:
        monitor = ProductionMonitor(tau_low=0.01, tau_high=0.38)

        # Called for every scored transaction
        monitor.log_decision(probability=0.87, decision=2)

        # Called when a human reviewer confirms a label
        monitor.log_confirmed(probability=0.87, true_label=1, decision=2)

        # Called periodically to get a performance snapshot
        snapshot = monitor.get_snapshot()
    """

    def __init__(
        self,
        tau_low:         float,
        tau_high:        float,
        review_window:   int = 100,
        confirmed_window:int = 500,
    ):
        self.tau_low          = tau_low
        self.tau_high         = tau_high
        self.review_window    = review_window
        self.confirmed_window = confirmed_window

        # Rolling buffer of the last N decisions (int: 0/1/2)
        self._decision_buffer: deque[int] = deque(maxlen=review_window)

        # Rolling buffer of confirmed cases
        self._confirmed: deque[ConfirmedCase] = deque(maxlen=confirmed_window)

        self._n_total = 0

    def log_decision(self, probability: float, decision: int) -> None:
        """Log a scored transaction decision (0=approve, 1=review, 2=block)."""
        self._decision_buffer.append(decision)
        self._n_total += 1

    def log_confirmed(
        self,
        probability: float,
        true_label: int,
        decision: int,
        timestamp: str = "",
    ) -> None:
        """Log a transaction whose true label has been confirmed by a reviewer."""
        self._confirmed.append(ConfirmedCase(
            probability   = probability,
            true_label    = true_label,
            model_decision= decision,
            timestamp     = timestamp,
        ))

    def rolling_review_rate(self) -> float:
        """Fraction of the last review_window decisions that were 'human review'."""
        buf = list(self._decision_buffer)
        if not buf:
            return 0.0
        return sum(1 for d in buf if d == 1) / len(buf)

    def get_snapshot(self) -> PerformanceSnapshot:
        confirmed = list(self._confirmed)

        fraud_cases = [c for c in confirmed if c.true_label == 1]
        legit_cases = [c for c in confirmed if c.true_label == 0]

        # Confirmed recall: fraud cases NOT auto-approved (model caught them)
        confirmed_recall = None
        if fraud_cases:
            caught = sum(1 for c in fraud_cases if c.model_decision != 0)
            confirmed_recall = caught / len(fraud_cases)

        # Confirmed precision: of auto-blocked transactions, what fraction were fraud
        confirmed_precision = None
        blocked = [c for c in confirmed if c.model_decision == 2]
        if blocked:
            true_blocks = sum(1 for c in blocked if c.true_label == 1)
            confirmed_precision = true_blocks / len(blocked)

        # Score-label correlation on confirmed cases
        score_label_corr = None
        if len(confirmed) >= 10:
            probs  = np.array([c.probability  for c in confirmed])
            labels = np.array([c.true_label   for c in confirmed])
            if probs.std() > 1e-6 and labels.std() > 1e-6:
                score_label_corr = float(np.corrcoef(probs, labels)[0, 1])

        return PerformanceSnapshot(
            confirmed_recall    = confirmed_recall,
            confirmed_precision = confirmed_precision,
            n_confirmed_fraud   = len(fraud_cases),
            n_confirmed_legit   = len(legit_cases),
            rolling_review_rate = self.rolling_review_rate(),
            score_label_corr    = score_label_corr,
            n_decisions_total   = self._n_total,
        )

    def should_retune_thresholds(self, r_max: float = 0.03) -> bool:
        """
        Returns True if the rolling review rate has exceeded R_max.
        This signals threshold re-tuning (not necessarily full retraining).
        """
        return self.rolling_review_rate() > r_max

    def should_retrain(self, recall_threshold: float = 0.75) -> bool:
        """
        Returns True if confirmed recall has dropped below recall_threshold.
        This signals that the model is systematically missing fraud —
        concept drift, not just threshold miscalibration.
        """
        snap = self.get_snapshot()
        if snap.confirmed_recall is None:
            return False   # Not enough confirmed data to decide
        return snap.confirmed_recall < recall_threshold
