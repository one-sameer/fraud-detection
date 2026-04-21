# Credit Card Fraud Detection — End-to-End Real-Time ML System

A production-grade fraud detection pipeline built on the [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Trains five models, calibrates their probabilities, combines them into a stacking ensemble, and serves real-time predictions through a FastAPI endpoint with a Streamlit monitoring dashboard.

---

## Results

| Metric | Value |
|---|---|
| Ensemble val AUPRC | 0.893 |
| Test AUPRC | 0.751 |
| Test Recall | 76.9% (40/52 fraud caught) |
| Test Precision | 63.5% |
| Test F1 | 0.696 |
| False positives | 23 (0.05% of legitimate transactions) |
| Human review rate | 0.14% of transactions |
| API latency (p99) | ~60ms on CPU |

The 14-point val→test AUPRC gap reflects genuine temporal distribution shift — fraud patterns in the later test period differ from training. The retraining loop in Stage 7 addresses this in production.

---

## Architecture

```
Transaction (raw JSON)
    │
    ▼
Preprocessor  (cyclical time encoding + RobustScaler for Amount)
    │
    ├── Logistic Regression ──┐
    ├── Random Forest ────────┤
    ├── XGBoost ──────────────┤  calibrated probabilities
    ├── LightGBM ─────────────┤
    └── MLP ──────────────────┘
                              │
                              ▼
                   Weighted average ensemble
                   (weights optimised on val set)
                              │
                              ▼
                   Cost-aware decision layer
                   ┌──────────────────────────┐
                   │  p < τ_low  → approve    │
                   │  τ_low ≤ p ≤ τ_high      │
                   │           → human review │
                   │  p > τ_high → block      │
                   └──────────────────────────┘
                              │
                              ▼
                   FastAPI /predict endpoint
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/fraud_detection.git
cd fraud_detection
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/creditcard.csv`.

### 3. Set cost parameters

Edit `configs/cost_config.yaml` before running Stage 5:

```yaml
costs:
  C_FN: 125.0   # cost of missing a fraud (approx. mean fraud amount)
  C_FP: 5.0     # cost of wrongly blocking a legitimate transaction
```

### 4. Run the pipeline

```bash
python stage1_preprocess.py --data data/creditcard.csv
python stage2_train_models.py
python stage3_calibrate.py
python stage4_ensemble.py        # ~30-60 min on CPU (5-fold OOF retraining)
python stage5_binary_threshold.py
python stage6_triclass_threshold.py
python stage7_monitoring_setup.py
python stage8_api.py             # smoke test + latency benchmark
```

### 5. Start the server and UI

```bash
# Terminal 1 — API (use --workers 1 on Windows)
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1

# Terminal 2 — Streamlit dashboard
streamlit run ui.py
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the Swagger UI.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/predict` | Score a transaction, returns 3-class decision |
| `GET` | `/health` | Liveness check, model version, active thresholds |
| `GET` | `/metrics` | Rolling monitoring metrics |
| `POST` | `/confirm` | Submit confirmed label from human reviewer |
| `POST` | `/admin/reload` | Hot-reload thresholds without restarting server |
| `GET` | `/docs` | Interactive Swagger UI |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0, "Amount": 149.62,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02
  }'
```

```json
{
  "fraud_probability": 0.000312,
  "decision": "legitimate",
  "decision_int": 0,
  "model_version": "v1.0",
  "review_rate_last_100": 0.0,
  "latency_ms": 58.3
}
```

### Updating thresholds without redeployment

```bash
# Edit configs/decision_config.yaml, then:
curl -X POST http://localhost:8000/admin/reload
```

---

## Configuration

### `configs/cost_config.yaml`

```yaml
costs:
  C_FN: 125.0        # Cost of a false negative (missed fraud)
  C_FP: 5.0          # Cost of a false positive (wrongly blocked legit)
  C_review: 2.0      # Cost per human review
  C_overflow: 500.0  # Penalty when review rate exceeds R_max
review:
  R_max: 0.03        # Max fraction of transactions sent to review (3%)
  rolling_window: 100
```

### `configs/decision_config.yaml`

Written automatically by Stages 5 and 6. Hot-reloadable at runtime.

```yaml
binary:
  threshold: 0.0248
three_class:
  tau_low: 0.0117     # below this → auto-approve
  tau_high: 0.6041    # above this → auto-block, between → human review
```

---

## Retraining

Two triggers in `src/monitoring/retrain.py`:

| Signal | Action |
|---|---|
| Score PSI > 0.2 or confirmed recall < 75% | Full retrain — all base models + ensemble |
| Score PSI 0.1–0.2 or weekly schedule | Meta-learner only retrain (fast) |

New models run in **shadow mode** — scored on every transaction but not returned to callers. Promoted to champion only when AUPRC on confirmed reviewer labels exceeds the current champion after a configurable shadow period.

---

## Project Structure

```
fraud_detection/
├── configs/
│   ├── cost_config.yaml
│   └── decision_config.yaml
├── data/
│   └── creditcard.csv            ← not in repo, download from Kaggle
├── src/
│   ├── api/app.py                ← FastAPI application
│   ├── calibration/              ← isotonic regression, cross-val calibration
│   ├── data/preprocessing.py     ← pipeline, temporal split, SMOTE
│   ├── decision/                 ← cost-aware threshold optimisation
│   ├── ensemble/                 ← stacking and weighted average ensembles
│   ├── evaluation/metrics.py     ← AUPRC, ECE, MCC, reliability diagrams
│   ├── models/                   ← LR, RF, XGBoost, LightGBM, MLP, Autoencoder
│   ├── monitoring/               ← PSI drift detection, retraining orchestrator
│   └── utils/mlflow_utils.py
├── stage1_preprocess.py
├── stage2_train_models.py
├── stage3_calibrate.py
├── stage4_ensemble.py
├── stage5_binary_threshold.py
├── stage6_triclass_threshold.py
├── stage7_monitoring_setup.py
├── stage8_api.py
├── ui.py
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**Temporal split** — train/val/test are split chronologically, not randomly. The test set contains only transactions from later in time, matching real production conditions and giving honest performance estimates.

**Probability calibration** — all models are post-hoc calibrated with cross-validated isotonic regression. This makes p=0.8 genuinely mean 80% of transactions at that score are fraud, which makes threshold tuning principled.

**Cost-aware thresholds** — τ* minimises `C_FN × false_negatives + C_FP × false_positives` rather than maximising F1. The ratio C_FN/C_FP = 25× reflects the business reality that missing fraud is far more costly than a false block.

**3-class decision layer** — the human review band captures genuinely ambiguous transactions rather than forcing a binary decision on uncertain cases. This simultaneously reduces false negatives and false positives compared to binary classification.

**Sensitivity-stable thresholds** — Stage 6 shows thresholds are flat across C_FN/C_FP ratios from 5× to 500×. The decision boundaries are driven by the bimodal score distribution, not the specific cost values chosen.

---

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions, 492 fraud (0.173%). Features V1–V28 are PCA-transformed and anonymised. `Time` and `Amount` are raw.
