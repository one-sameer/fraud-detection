"""
ui.py
──────
Streamlit dashboard for the fraud detection API.

Two modes:
  Auto   — sends one transaction per rerun cycle, plots live probability stream.
  Manual — set all feature values explicitly, inspect the full response.

Run (API must be running first):
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1
    streamlit run ui.py
"""

import time
import random
import requests
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


API_BASE = "http://localhost:8000"


# ── Load real examples from creditcard.csv ────────────────────────────────────

@st.cache_data
def load_real_examples():
    try:
        csv_path = Path("data/creditcard.csv")
        if not csv_path.exists():
            return [], [], f"data/creditcard.csv not found at {csv_path.resolve()}"

        df = pd.read_csv(csv_path).sort_values("Time").reset_index(drop=True)
        n = len(df)
        test_start = int(n * 0.85)
        test_df = df.iloc[test_start:].reset_index(drop=True)

        v_cols = [f"V{i}" for i in range(1, 29)]
        feature_cols = ["Time", "Amount"] + v_cols

        fraud_rows = test_df[test_df["Class"] == 1][feature_cols]
        legit_rows = test_df[test_df["Class"] == 0][feature_cols].sample(
            min(500, len(test_df[test_df["Class"] == 0])), random_state=42
        )

        def to_payload(row):
            return {col: float(row[col]) for col in feature_cols}

        fraud_ex = [to_payload(r) for _, r in fraud_rows.iterrows()]
        legit_ex = [to_payload(r) for _, r in legit_rows.iterrows()]
        return fraud_ex, legit_ex, None

    except Exception as e:
        return [], [], str(e)


_res = load_real_examples()
FRAUD_EXAMPLES, LEGIT_EXAMPLES, _LOAD_ERROR = _res[0], _res[1], _res[2]

# V-feature stats for manual mode slider bounds (mu, sigma)
V_STATS = {
    "V1":  (0.0, 1.96), "V2":  (0.0, 1.65), "V3":  (0.0, 1.52),
    "V4":  (0.0, 1.42), "V5":  (0.0, 1.38), "V6":  (0.0, 1.33),
    "V7":  (0.0, 1.24), "V8":  (0.0, 1.19), "V9":  (0.0, 1.10),
    "V10": (0.0, 1.09), "V11": (0.0, 1.02), "V12": (0.0, 1.00),
    "V13": (0.0, 1.00), "V14": (0.0, 0.96), "V15": (0.0, 0.92),
    "V16": (0.0, 0.88), "V17": (0.0, 0.85), "V18": (0.0, 0.84),
    "V19": (0.0, 0.81), "V20": (0.0, 0.77), "V21": (0.0, 0.73),
    "V22": (0.0, 0.73), "V23": (0.0, 0.62), "V24": (0.0, 0.61),
    "V25": (0.0, 0.52), "V26": (0.0, 0.48), "V27": (0.0, 0.40),
    "V28": (0.0, 0.33),
}

DECISION_COLORS = {
    "legitimate":   "#2ECC71",
    "human_review": "#F39C12",
    "fraud":        "#E74C3C",
}


# ── API helpers ───────────────────────────────────────────────────────────────

def check_api_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def send_transaction(payload):
    try:
        r = requests.post(f"{API_BASE}/predict", json=payload, timeout=10)
        return r.json() if r.status_code == 200 else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_server_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=2)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def random_payload(fraud_bias=0.0):
    """
    fraud_bias >= 0.5 → draw from real fraud examples.
    fraud_bias <  0.5 → draw from real legit examples.
    Falls back to synthetic if CSV not available.
    """
    if fraud_bias >= 0.5 and FRAUD_EXAMPLES:
        return random.choice(FRAUD_EXAMPLES)
    if LEGIT_EXAMPLES:
        return random.choice(LEGIT_EXAMPLES)
    # Synthetic fallback
    payload = {
        "Time":   random.uniform(0, 172792),
        "Amount": max(0.01, float(np.random.lognormal(3.5, 1.5))),
    }
    for feat, (mu, sigma) in V_STATS.items():
        payload[feat] = float(np.random.normal(mu, sigma))
    return payload


# ── Session state ─────────────────────────────────────────────────────────────

def _init():
    defaults = {
        "history":         [],
        "total_sent":      0,
        "counts":          {"legitimate": 0, "human_review": 0, "fraud": 0},
        "auto_running":    False,
        "use_for_retrain": False,
        "retrain_buffer":  [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


def record(resp, payload, known_label=None):
    resp["_amount"] = payload.get("Amount", 0)
    st.session_state.history.append(resp)
    st.session_state.total_sent += 1
    d = resp.get("decision", "legitimate")
    st.session_state.counts[d] = st.session_state.counts.get(d, 0) + 1
    if st.session_state.use_for_retrain and known_label is not None:
        st.session_state.retrain_buffer.append({
            "fraud_probability": resp.get("fraud_probability", 0),
            "true_label":        known_label,
            "decision_int":      resp.get("decision_int", 0),
        })


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.decision-badge {
  display:inline-block; padding:5px 16px; border-radius:16px;
  font-size:15px; font-weight:600; margin:4px 0;
}
.legit  { background:#d5f5e3; color:#1a7a41; }
.review { background:#fdebd0; color:#b7770d; }
.fraud  { background:#fadbd8; color:#a93226; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Fraud Detection")
    st.divider()

    health = check_api_health()
    if health:
        st.success(f"API online — {health.get('model_version','?')}")
        st.caption(
            f"tau_low={health.get('tau_low','?')}  "
            f"tau_high={health.get('tau_high','?')}"
        )
    else:
        st.error("API offline")
        st.code("uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1")

    st.divider()

    if _LOAD_ERROR:
        st.warning(f"Real examples not loaded:\n{_LOAD_ERROR}")
    else:
        st.success(
            f"{len(FRAUD_EXAMPLES)} fraud + {len(LEGIT_EXAMPLES)} legit "
            f"examples loaded from test set"
        )

    st.divider()
    mode = st.radio("Mode", ["Auto", "Manual"], horizontal=True)
    st.divider()

    if mode == "Auto":
        fraud_bias = st.slider(
            "Fraud bias",
            0.0, 1.0, 0.0, 0.05,
            help="Below 0.5: draw legitimate examples. At or above 0.5: draw fraud examples."
        )
        speed_ms = st.slider("Delay between requests (ms)", 200, 3000, 800, 100)
        n_show   = st.slider("Transactions to display", 20, 300, 100, 10)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Start", width='stretch',
                         disabled=st.session_state.auto_running):
                st.session_state.auto_running = True
                st.rerun()
        with c2:
            if st.button("Stop", width='stretch',
                         disabled=not st.session_state.auto_running):
                st.session_state.auto_running = False
                st.rerun()

        if st.session_state.auto_running:
            st.info("Streaming — press Stop to pause")
        else:
            st.caption("Press Start to begin")

    st.divider()
    st.subheader("Retraining")
    st.session_state.use_for_retrain = st.toggle(
        "Buffer simulation data for retraining",
        value=st.session_state.use_for_retrain,
        help=(
            "When on, transactions sent via this UI are buffered as retraining "
            "candidates. Disable this when running fraud-heavy simulations — "
            "over-representing fraud artificially would bias a retrained model "
            "toward higher false positive rates on real traffic."
        )
    )
    if st.session_state.retrain_buffer:
        st.caption(f"{len(st.session_state.retrain_buffer)} samples buffered")
        if st.button("Clear retrain buffer"):
            st.session_state.retrain_buffer = []
            st.rerun()

    st.divider()
    st.subheader("Server metrics")
    if health:
        m = get_server_metrics()
        if m:
            st.metric("Review rate (live)",
                      f"{m.get('rolling_review_rate',0)*100:.2f}%")
            rc = m.get("confirmed_recall")
            st.metric("Confirmed recall", f"{rc:.4f}" if rc else "—")
            st.metric("Total decisions", f"{m.get('n_decisions_total',0):,}")
            if m.get("retrain_flag"):
                st.warning("Retraining recommended")
            if m.get("threshold_retune_flag"):
                st.info("Threshold re-tune suggested")

    if st.button("Clear session history", width='stretch'):
        st.session_state.history  = []
        st.session_state.total_sent = 0
        st.session_state.counts   = {"legitimate": 0, "human_review": 0, "fraud": 0}
        st.rerun()


# ─── AUTO MODE ────────────────────────────────────────────────────────────────

if mode == "Auto":
    st.header("Auto mode — live transaction stream")

    if not health:
        st.error("Start the API server first.")
        st.stop()

    t  = st.session_state.total_sent
    co = st.session_state.counts
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total sent",  f"{t:,}")
    c2.metric("Legitimate",  f"{co['legitimate']:,}",
              f"{co['legitimate']/max(t,1)*100:.1f}%")
    c3.metric("Review",      f"{co['human_review']:,}",
              f"{co['human_review']/max(t,1)*100:.1f}%")
    c4.metric("Fraud",       f"{co['fraud']:,}",
              f"{co['fraud']/max(t,1)*100:.1f}%")

    st.divider()

    hist = st.session_state.history[-n_show:]
    if hist:
        import altair as alt

        df_c = pd.DataFrame([{
            "idx":        i,
            "probability": h.get("fraud_probability", 0),
            "decision":    h.get("decision", "legitimate"),
            "amount":      h.get("_amount", 0),
            "latency_ms":  h.get("latency_ms", 0),
        } for i, h in enumerate(hist)])

        tau_lo = health.get("tau_low",  0.01)
        tau_hi = health.get("tau_high", 0.99)

        color_scale = alt.Scale(
            domain=["legitimate", "human_review", "fraud"],
            range=["#2ECC71", "#F39C12", "#E74C3C"]
        )
        band_r = alt.Chart(pd.DataFrame({"y1": [tau_lo], "y2": [tau_hi]})).mark_rect(
            opacity=0.07, color="#F39C12"
        ).encode(y="y1:Q", y2="y2:Q")

        band_f = alt.Chart(pd.DataFrame({"y1": [tau_hi], "y2": [1.0]})).mark_rect(
            opacity=0.05, color="#E74C3C"
        ).encode(y="y1:Q", y2="y2:Q")

        rule_lo = alt.Chart(pd.DataFrame({"y": [tau_lo]})).mark_rule(
            color="#F39C12", strokeDash=[4, 3], strokeWidth=1.2
        ).encode(y="y:Q")

        rule_hi = alt.Chart(pd.DataFrame({"y": [tau_hi]})).mark_rule(
            color="#E74C3C", strokeDash=[4, 3], strokeWidth=1.2
        ).encode(y="y:Q")

        base  = alt.Chart(df_c).encode(x=alt.X("idx:Q", title="Transaction #"))
        line  = base.mark_line(opacity=0.2, color="#4A90D9", strokeWidth=1).encode(
            y="probability:Q"
        )
        dots  = base.mark_circle(size=55, opacity=0.9).encode(
            y=alt.Y("probability:Q", title="Fraud probability",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("decision:N", scale=color_scale,
                            legend=alt.Legend(title="Decision")),
            tooltip=["probability:Q", "decision:N", "amount:Q", "latency_ms:Q"],
        )
        chart = (band_r + band_f + rule_lo + rule_hi + line + dots).properties(
            height=300, title="Fraud probability stream"
        ).configure_view(strokeWidth=0)

        st.altair_chart(chart, width='stretch')

        last  = hist[-1]
        d     = last.get("decision", "legitimate")
        p     = last.get("fraud_probability", 0)
        amt   = last.get("_amount", 0)
        lat   = last.get("latency_ms", 0)
        badge = {"legitimate": "legit", "human_review": "review", "fraud": "fraud"}[d]
        st.markdown(
            f"Latest: "
            f'<span class="decision-badge {badge}">{d.replace("_"," ").title()}</span>'
            f" &nbsp; p = <b>{p:.5f}</b> &nbsp;|&nbsp; "
            f"Amount: <b>${amt:.2f}</b> &nbsp;|&nbsp; "
            f"Latency: <b>{lat:.0f} ms</b>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Press Start in the sidebar to begin streaming.")

    if st.session_state.history:
        with st.expander(f"Transaction history ({len(st.session_state.history)} rows)"):
            df_h = pd.DataFrame([{
                "#":           i + 1,
                "probability": round(h.get("fraud_probability", 0), 5),
                "decision":    h.get("decision", ""),
                "amount":      round(h.get("_amount", 0), 2),
                "latency_ms":  h.get("latency_ms", 0),
            } for i, h in enumerate(st.session_state.history)])
            st.dataframe(df_h, width='stretch', height=260)

    # Single-step streaming: one request per rerun
    if st.session_state.auto_running:
        payload    = random_payload(fraud_bias=fraud_bias)
        true_label = 1 if (fraud_bias >= 0.5 and FRAUD_EXAMPLES) else 0
        resp       = send_transaction(payload)
        if resp and "error" not in resp:
            record(resp, payload, known_label=true_label)
        time.sleep(speed_ms / 1000.0)
        st.rerun()


# ─── MANUAL MODE ──────────────────────────────────────────────────────────────

else:
    st.header("Manual mode — set feature values")

    if not health:
        st.error("Start the API server first.")

    st.caption(
        "V1–V28 are PCA-transformed anonymised features. "
        "Typical legitimate values cluster near 0. "
        "Strongly negative V14, V12, V3, V10 correlate with fraud."
    )

    with st.form("manual_form"):
        col_t, col_a = st.columns(2)
        with col_t:
            time_val   = st.number_input("Time (seconds)", value=406.0, step=1.0)
        with col_a:
            amount_val = st.number_input("Amount ($)", value=50.0,
                                          min_value=0.01, step=1.0)

        st.divider()
        st.subheader("V features")

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            fill_legit = st.form_submit_button("Fill — random legitimate")
        with fc2:
            fill_fraud = st.form_submit_button("Fill — random fraud-like")
        with fc3:
            send_btn   = st.form_submit_button("Send transaction", type="primary")

        if "manual_vals" not in st.session_state:
            st.session_state.manual_vals = {f: 0.0 for f in V_STATS}

        if fill_legit:
            if LEGIT_EXAMPLES:
                ex = random.choice(LEGIT_EXAMPLES)
                st.session_state.manual_vals = {f: ex.get(f, 0.0) for f in V_STATS}
            else:
                st.session_state.manual_vals = {
                    f: float(np.random.normal(mu, sigma))
                    for f, (mu, sigma) in V_STATS.items()
                }

        if fill_fraud:
            if FRAUD_EXAMPLES:
                ex = random.choice(FRAUD_EXAMPLES)
                st.session_state.manual_vals = {f: ex.get(f, 0.0) for f in V_STATS}
            else:
                st.session_state.manual_vals = {
                    f: float(np.random.normal(mu - 3*sigma, sigma))
                    for f, (mu, sigma) in V_STATS.items()
                }

        cols   = st.columns(4)
        v_vals = {}
        for i, (feat, (mu, sigma)) in enumerate(V_STATS.items()):
            lo      = float(mu - 15 * sigma)
            hi      = float(mu + 15 * sigma)
            default = float(np.clip(st.session_state.manual_vals.get(feat, 0.0), lo, hi))
            v_vals[feat] = cols[i % 4].number_input(
                feat,
                value=round(default, 4),
                min_value=lo,
                max_value=hi,
                step=round(sigma * 0.1, 4),
                format="%.4f",
            )

        known_label = st.selectbox(
            "True label (optional — for retraining buffer)",
            options=[None, 0, 1],
            format_func=lambda x: "Unknown" if x is None
                                  else ("Legitimate" if x == 0 else "Fraud"),
        )

        if send_btn:
            payload = {"Time": time_val, "Amount": amount_val, **v_vals}
            with st.spinner("Scoring..."):
                resp = send_transaction(payload)

            if resp and "error" not in resp:
                record(resp, payload, known_label=known_label)
                d    = resp.get("decision", "legitimate")
                p    = resp.get("fraud_probability", 0)
                lat  = resp.get("latency_ms", 0)
                rr   = resp.get("review_rate_last_100", 0)
                col  = DECISION_COLORS.get(d, "#888")
                badge = {"legitimate": "legit",
                         "human_review": "review",
                         "fraud": "fraud"}[d]

                st.markdown(f"""
                <div style="border:2px solid {col};border-radius:10px;
                            padding:16px 22px;margin:10px 0">
                  <span class="decision-badge {badge}">
                    {d.replace("_"," ").title()}
                  </span>
                  <div style="display:flex;gap:36px;font-size:14px;margin-top:10px">
                    <div><b>Fraud probability</b><br>
                         <span style="font-size:22px;font-weight:700">{p:.5f}</span>
                    </div>
                    <div><b>Model</b><br>{resp.get("model_version","?")}</div>
                    <div><b>Latency</b><br>{lat:.0f} ms</div>
                    <div><b>Review rate (last 100)</b><br>{rr*100:.2f}%</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                tau_lo = health.get("tau_low",  0.01) if health else 0.01
                tau_hi = health.get("tau_high", 0.99) if health else 0.99
                st.progress(min(p * 3, 1.0),
                            text=f"Probability (x3 zoom): {p:.5f}")
                st.caption(
                    f"Bands:  p < {tau_lo} = legitimate  |  "
                    f"{tau_lo} to {tau_hi} = review  |  "
                    f"p > {tau_hi} = fraud"
                )
            elif resp:
                st.error(f"API error: {resp.get('error')}")

    if st.session_state.history:
        st.divider()
        st.subheader(f"Session history ({len(st.session_state.history)} transactions)")
        df_h = pd.DataFrame([{
            "#":           i + 1,
            "decision":    h.get("decision", ""),
            "probability": round(h.get("fraud_probability", 0), 5),
            "amount":      round(h.get("_amount", 0), 2),
            "latency_ms":  h.get("latency_ms", 0),
        } for i, h in enumerate(st.session_state.history)])

        def _col(val):
            c = {"legitimate": "#d5f5e3",
                 "human_review": "#fdebd0",
                 "fraud": "#fadbd8"}
            return f"background-color: {c.get(val,'')}"

        st.dataframe(
            df_h.style.map(_col, subset=["decision"]),
            width='stretch', height=240
        )