"""
src/api/app.py
───────────────
FastAPI inference server.

Endpoints:
    POST /predict           ← main inference endpoint
    GET  /health            ← liveness + model version + current thresholds
    GET  /metrics           ← rolling monitoring metrics
    POST /confirm           ← submit a confirmed label from human review
    POST /admin/reload      ← hot-reload thresholds from decision_config.yaml
    GET  /admin/thresholds  ← inspect current threshold values

Inference pipeline (per request):
    1. Validate input (pydantic)
    2. Build feature array → apply preprocessor → ensemble.predict_proba()
    3. classify_transaction(p, τ_low, τ_high) → decision
    4. log_decision() to the in-memory ProductionMonitor
    5. Return { fraud_probability, decision, model_version, review_rate }

Performance target: p99 < 50ms
    - Preprocessor + ensemble load once at startup (lifespan context)
    - Thresholds are plain floats in memory, not file reads per request
    - Hot-reload is triggered explicitly via /admin/reload, never per-request
    - Model is never reloaded mid-request (asyncio lock protects reload)
"""

from __future__ import annotations

import asyncio
import time
import yaml
import numpy as np
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── project imports ────────────────────────────────────────────────────────────
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessing import load_preprocessor, FEATURE_COLS
from src.decision.threshold_optimizer import classify_transaction
from src.monitoring.performance import ProductionMonitor

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR         = Path(__file__).parent.parent.parent
DECISION_CONFIG  = BASE_DIR / "configs" / "decision_config.yaml"
PREPROCESSOR_PATH= BASE_DIR / "outputs" / "artifacts" / "preprocessor.joblib"
ENSEMBLE_PATH    = BASE_DIR / "outputs" / "models"    / "production_ensemble.joblib"
REGISTRY_PATH    = BASE_DIR / "outputs" / "models"    / "model_registry.json"

# ─── App state ────────────────────────────────────────────────────────────────

class AppState:
    """Single mutable state object — avoids module-level globals."""
    preprocessor    = None
    ensemble        = None
    tau_low:  float = 0.01
    tau_high: float = 0.99
    tau_binary: float = 0.5
    model_version: str = "unknown"
    monitor: ProductionMonitor = None
    reload_lock: asyncio.Lock = None

    # Rolling decision log for review rate (list of ints: 0/1/2)
    _decision_log: list[int] = []

state = AppState()


# ─── Startup / shutdown ───────────────────────────────────────────────────────

def _load_thresholds() -> tuple[float, float, float]:
    with open(DECISION_CONFIG) as f:
        cfg = yaml.safe_load(f)
    tau_binary = cfg["binary"]["threshold"] or 0.5
    tau_low    = cfg["three_class"]["tau_low"]  or 0.01
    tau_high   = cfg["three_class"]["tau_high"] or 0.99
    return float(tau_binary), float(tau_low), float(tau_high)


def _load_model_version() -> str:
    try:
        import json
        with open(REGISTRY_PATH) as f:
            records = json.load(f)
        for r in reversed(records):
            if r["status"] == "champion":
                return r["version"]
    except Exception:
        pass
    return "v1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models once at startup. Shuts down cleanly."""
    print("[startup] Loading preprocessor...")
    state.preprocessor = load_preprocessor(str(PREPROCESSOR_PATH))

    print("[startup] Loading production ensemble...")
    state.ensemble = joblib.load(str(ENSEMBLE_PATH))

    print("[startup] Loading thresholds...")
    state.tau_binary, state.tau_low, state.tau_high = _load_thresholds()

    state.model_version = _load_model_version()
    state.monitor       = ProductionMonitor(
        tau_low=state.tau_low, tau_high=state.tau_high
    )
    state.reload_lock   = asyncio.Lock()
    state._decision_log = []

    print(f"[startup] Ready  model={state.model_version}  "
          f"τ_low={state.tau_low}  τ_high={state.tau_high}")
    yield
    print("[shutdown] Goodbye.")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Fraud Detection API",
    description = "Real-time credit card fraud detection with 3-class decision output.",
    version     = "1.0.0",
    lifespan    = lifespan,
)


# ─── Request / response schemas ───────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    Raw transaction features as received from the payment system.
    Mirrors the ULB dataset schema exactly.
    """
    Time:   float = Field(..., description="Seconds elapsed from dataset start")
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    V1:  float; V2:  float; V3:  float; V4:  float; V5:  float
    V6:  float; V7:  float; V8:  float; V9:  float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0, "Amount": 149.62,
                "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
                "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
                "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
                "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
                "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
                "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
                "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
            }
        }


class PredictResponse(BaseModel):
    fraud_probability:      float
    decision:               str     # "legitimate" | "human_review" | "fraud"
    decision_int:           int     # 0 | 1 | 2
    model_version:          str
    review_rate_last_100:   float
    latency_ms:             float


class ConfirmRequest(BaseModel):
    transaction_id: str
    fraud_probability: float
    decision_int: int
    true_label: int = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    status:        str
    model_version: str
    tau_low:       float
    tau_high:      float
    tau_binary:    float
    uptime_s:      float


# ─── Utility ──────────────────────────────────────────────────────────────────

_startup_time = time.time()

def _request_to_df(req: TransactionRequest) -> "pd.DataFrame":
    """Converts a TransactionRequest to a single-row DataFrame with named columns.
    The preprocessor was fitted on a DataFrame so it requires column names."""
    row = {f: getattr(req, f) for f in FEATURE_COLS}
    return pd.DataFrame([row])


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(request: TransactionRequest):
    """
    Score a single transaction and return a 3-class decision.

    The pipeline:
        raw features → preprocessor → ensemble → probability → threshold → decision
    """
    t0 = time.perf_counter()

    # Build raw feature DataFrame (preprocessor needs named columns)
    raw = _request_to_df(request)

    # Apply preprocessing pipeline (same object fitted in Stage 1)
    try:
        X = state.preprocessor.transform(raw).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # Score
    try:
        p = float(state.ensemble.predict_proba(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

    # Classify
    decision_int, decision_label = classify_transaction(p, state.tau_low, state.tau_high)

    # Log to monitor
    state.monitor.log_decision(probability=p, decision=decision_int)
    state._decision_log.append(decision_int)

    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        fraud_probability    = round(p, 6),
        decision             = decision_label,
        decision_int         = decision_int,
        model_version        = state.model_version,
        review_rate_last_100 = round(state.monitor.rolling_review_rate(), 4),
        latency_ms           = round(latency_ms, 2),
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — returns model version and active thresholds."""
    return HealthResponse(
        status        = "ok",
        model_version = state.model_version,
        tau_low       = state.tau_low,
        tau_high      = state.tau_high,
        tau_binary    = state.tau_binary,
        uptime_s      = round(time.time() - _startup_time, 1),
    )


@app.get("/metrics")
async def metrics():
    """
    Rolling monitoring metrics from the in-memory ProductionMonitor.
    Intended for a Prometheus scraper or a dashboard poll.
    """
    snap = state.monitor.get_snapshot()
    return {
        "rolling_review_rate":   round(snap.rolling_review_rate, 4),
        "confirmed_recall":      round(snap.confirmed_recall, 4)
                                 if snap.confirmed_recall is not None else None,
        "confirmed_precision":   round(snap.confirmed_precision, 4)
                                 if snap.confirmed_precision is not None else None,
        "n_confirmed_fraud":     snap.n_confirmed_fraud,
        "n_confirmed_legit":     snap.n_confirmed_legit,
        "score_label_corr":      round(snap.score_label_corr, 4)
                                 if snap.score_label_corr is not None else None,
        "n_decisions_total":     snap.n_decisions_total,
        "threshold_retune_flag": state.monitor.should_retune_thresholds(),
        "retrain_flag":          state.monitor.should_retrain(),
    }


@app.post("/confirm")
async def confirm_label(body: ConfirmRequest):
    """
    Accepts a confirmed fraud/legit label from a human reviewer.
    Feeds the label into the ProductionMonitor for recall/precision tracking.
    """
    state.monitor.log_confirmed(
        probability = body.fraud_probability,
        true_label  = body.true_label,
        decision    = body.decision_int,
        timestamp   = datetime.utcnow().isoformat(),
    )
    snap = state.monitor.get_snapshot()
    return {
        "confirmed":          True,
        "n_confirmed_total":  snap.n_confirmed_fraud + snap.n_confirmed_legit,
        "confirmed_recall":   snap.confirmed_recall,
        "retrain_flag":       state.monitor.should_retrain(),
    }


@app.post("/admin/reload")
async def reload_thresholds():
    """
    Hot-reloads τ_low and τ_high from configs/decision_config.yaml.

    Use this after manually updating the config file, or after running
    stage5/stage6 to apply new optimised thresholds — no server restart needed.

    Protected by an asyncio lock so a reload never races with an inference.
    """
    async with state.reload_lock:
        old_low, old_high = state.tau_low, state.tau_high
        try:
            state.tau_binary, state.tau_low, state.tau_high = _load_thresholds()
            # Update the monitor's thresholds too
            state.monitor.tau_low  = state.tau_low
            state.monitor.tau_high = state.tau_high
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Reload failed: {e}")

    return {
        "reloaded":       True,
        "tau_low_old":    old_low,
        "tau_high_old":   old_high,
        "tau_low_new":    state.tau_low,
        "tau_high_new":   state.tau_high,
    }


@app.get("/admin/thresholds")
async def get_thresholds():
    """Inspect currently active threshold values without reloading."""
    return {
        "tau_binary": state.tau_binary,
        "tau_low":    state.tau_low,
        "tau_high":   state.tau_high,
        "model_version": state.model_version,
    }