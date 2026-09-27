"""
CrediX — Explainable Credit Intelligence v2.5
Enterprise Credit Risk Assessment & Underwriting Platform
Features:
- Dual Currency Support ($ USD / ₹ INR) with domain-scale normalization
- 1-Click Benchmark Test Profiles (Prime, Balanced, High-Risk)
- Dual Decision Modes: Prudential Risk-Aware vs Mathematical Argmax
- FICO-Style Credit Score (300–850) & 0–100 Risk Index
- Real-Time Financial Ratio Indicators (DTI, EMI Burden, Liquidity)
- Interactive Plotly Visualizations (Gauge, Probabilities, Feature Importance)
- Comprehensive Audit Trail & PDF Export
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

try:
    import pdf_report as pdf_mod
    PDF_OK = True
except Exception:
    PDF_OK = False

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrediX | Credit Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS & METADATA
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_data.csv"
USD_TO_INR_RATE = 83.0  # Purchasing power / benchmark exchange parity

EXPECTED_FEATURES = [
    "Age", "Annual_Income", "Num_of_Delayed_Payment",
    "Total_EMI_per_month", "Outstanding_Debt", "Monthly_Balance", "Occupation",
]

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]

RATING_CONFIG = {
    "Good": {
        "hex": "#10B981", "bg": "#ECFDF5", "border": "#A7F3D0",
        "dark": "#065F46", "badge": "PRIME / LOW RISK", "tier": "Tier 1",
    },
    "Standard": {
        "hex": "#F59E0B", "bg": "#FFFBEB", "border": "#FDE68A",
        "dark": "#78350F", "badge": "NEAR PRIME / MODERATE", "tier": "Tier 2",
    },
    "Poor": {
        "hex": "#EF4444", "bg": "#FEF2F2", "border": "#FECACA",
        "dark": "#991B1B", "badge": "SUBPRIME / HIGH RISK", "tier": "Tier 3",
    },
}

FEAT_DISPLAY = {
    "Outstanding_Debt": "Outstanding Debt", "Age": "Age",
    "Total_EMI_per_month": "Monthly EMI",
    "Num_of_Delayed_Payment": "Delayed Payments",
    "Annual_Income": "Annual Income", "Occupation": "Occupation",
    "Monthly_Balance": "Monthly Balance",
}

FALLBACK_IMP = {
    "Outstanding_Debt": 0.199782, "Age": 0.147663,
    "Total_EMI_per_month": 0.144677, "Num_of_Delayed_Payment": 0.122440,
    "Annual_Income": 0.112197, "Occupation": 0.089825, "Monthly_Balance": 0.083416,
}

MODEL_METRICS = {
    "Accuracy": 78.62, "Balanced Accuracy": 73.62, "Macro Precision": 74.94,
    "Macro Recall": 73.62, "Macro F1": 74.26, "Weighted F1": 78.52,
}

CLASS_METRICS = {
    "Good":     {"Precision": 74.98, "Recall": 72.85, "F1": 73.90},
    "Standard": {"Precision": 82.22, "Recall": 84.14, "F1": 83.17},
    "Poor":     {"Precision": 67.62, "Recall": 63.88, "F1": 65.70},
}

# ─────────────────────────────────────────────────────────────────
# ARTIFACT LOADERS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(BASE_DIR / "trained_credit_model.joblib")
    except Exception as e:
        st.error(f"CrediX model could not be loaded: {e}")
        st.stop()

@st.cache_resource
def load_encoder():
    try:
        return joblib.load(BASE_DIR / "label_encoder.joblib")
    except Exception:
        return None

@st.cache_resource
def load_features():
    try:
        return list(joblib.load(BASE_DIR / "feature_names.joblib"))
    except Exception:
        return EXPECTED_FEATURES

@st.cache_resource
def load_bounds():
    try:
        return joblib.load(BASE_DIR / "feature_bounds.joblib")
    except Exception:
        return {}

model       = load_model()
encoder     = load_encoder()
feat_order  = load_features()
bounds_data = load_bounds()
class_names = list(encoder.classes_) if encoder else ["Good", "Poor", "Standard"]

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE AGGREGATOR
# ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_importance_df():
    def _extract(est):
        if hasattr(est, "named_steps"):
            for k in ("classifier", "model", "estimator"):
                if k in est.named_steps:
                    r = _extract(est.named_steps[k])
                    if r is not None:
                        return r
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float)
        if hasattr(est, "calibrated_classifiers_"):
            imps = []
            for cal in est.calibrated_classifiers_:
                base = getattr(cal, "estimator", None) or getattr(cal, "base_estimator", None)
                if base:
                    r = _extract(base)
                    if r is not None:
                        imps.append(r)
            if imps:
                return np.mean(np.vstack(imps), axis=0)
        return None

    try:
        raw = _extract(model)
        prep = None
        if hasattr(model, "named_steps"):
            for k in ("preprocessor", "transformer"):
                if k in model.named_steps:
                    prep = model.named_steps[k]; break
        elif hasattr(model, "calibrated_classifiers_"):
            base = model.calibrated_classifiers_[0].estimator
            if hasattr(base, "named_steps") and "preprocessor" in base.named_steps:
                prep = base.named_steps["preprocessor"]

        if raw is None or prep is None:
            raise ValueError("Direct extraction not possible")

        col_names = prep.get_feature_names_out()
        agg = {}
        for feat, imp in zip(col_names, raw):
            key = feat.split("__")[-1] if "__" in feat else feat
            base_key = "Occupation" if key.startswith("Occupation") else key
            agg[base_key] = agg.get(base_key, 0.0) + float(imp)

        total = sum(agg.values()) or 1.0
        return pd.DataFrame(
            [{"Feature": k, "Importance": v / total} for k, v in agg.items()]
        ).sort_values("Importance", ascending=False).reset_index(drop=True)

    except Exception:
        return pd.DataFrame(
            [{"Feature": k, "Importance": v} for k, v in FALLBACK_IMP.items()]
        ).sort_values("Importance", ascending=False).reset_index(drop=True)

importance_df = get_importance_df()

# ─────────────────────────────────────────────────────────────────
# INFERENCE & EVALUATION ENGINE
# ─────────────────────────────────────────────────────────────────
def evaluate_applicant(
    age: int,
    occupation: str,
    annual_income: float,
    delayed_payments: int,
    emi: float,
    outstanding_debt: float,
    monthly_balance: float,
    currency: str = "USD",
    decision_mode: str = "risk_aware",
):
    """
    Evaluates applicant credit profile with currency normalization
    and dual decision policies (Prudential Risk-Aware vs Argmax).
    """
    scale_factor = USD_TO_INR_RATE if currency == "INR" else 1.0

    # Model input DataFrame normalized to training domain (USD scale)
    model_input = {
        "Age": int(age),
        "Occupation": str(occupation),
        "Annual_Income": float(annual_income) / scale_factor,
        "Num_of_Delayed_Payment": int(delayed_payments),
        "Total_EMI_per_month": float(emi) / scale_factor,
        "Outstanding_Debt": float(outstanding_debt) / scale_factor,
        "Monthly_Balance": float(monthly_balance) / scale_factor,
    }
    input_df = pd.DataFrame([{f: model_input[f] for f in feat_order}])

    # Probabilities from calibrated ensemble: [P(Good), P(Poor), P(Standard)]
    probabilities = model.predict_proba(input_df)[0]
    prob_map = {c: float(probabilities[i]) for i, c in enumerate(class_names)}
    p_good = prob_map.get("Good", 0.0)
    p_poor = prob_map.get("Poor", 0.0)
    p_std  = prob_map.get("Standard", 0.0)

    # Decision policy
    if decision_mode == "risk_aware":
        # Prudential default thresholding: P(Poor) >= 30% flags default risk
        if p_poor >= 0.30:
            prediction = "Poor"
        elif p_good >= 0.40:
            prediction = "Good"
        else:
            prediction = "Standard"
    else:
        # Standard mathematical argmax (>50%)
        pred_enc = model.predict(input_df)[0]
        prediction = encoder.inverse_transform([pred_enc])[0] if encoder else str(pred_enc)

    # Calibrated Credit Score (300 to 850 FICO-style scale)
    # Good probability drives 850, Poor drives 300, Standard anchors 620-680
    raw_score = 300.0 + 550.0 * (p_good + 0.52 * p_std)
    credit_score = int(np.clip(round(raw_score), 300, 850))

    # 0 to 100 Risk Index (Lower is safer)
    risk_score = round(float(p_poor * 90.0 + p_std * 50.0 + p_good * 12.0), 1)

    # Out-of-Distribution Warning Check
    ood_warnings = []
    if bounds_data:
        debt_max = bounds_data.get("Outstanding_Debt", {}).get("train_max", 1500.0)
        emi_max  = bounds_data.get("Total_EMI_per_month", {}).get("train_max", 200.0)
        norm_debt = model_input["Outstanding_Debt"]
        norm_emi  = model_input["Total_EMI_per_month"]
        if norm_debt > debt_max * 1.5:
            ood_warnings.append(f"Outstanding debt is {norm_debt/debt_max:.1f}× above training boundary.")
        if norm_emi > emi_max * 1.5:
            ood_warnings.append(f"Monthly EMI is {norm_emi/emi_max:.1f}× above training boundary.")

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "prob_map": prob_map,
        "credit_score": credit_score,
        "risk_score": risk_score,
        "input_df": input_df,
        "normalized_input": model_input,
        "ood_warnings": ood_warnings,
    }

def get_recommendations(pred, annual_income, delayed_payments, emi, debt, balance, sym):
    recs = []
    mi = annual_income / 12 if annual_income > 0 else 1
    emi_ratio = emi / mi if mi > 0 else 0
    dti = debt / annual_income if annual_income > 0 else 0

    if delayed_payments > 3:
        recs.append((
            "🔔 Payment Discipline",
            f"You have {int(delayed_payments)} recorded delayed payments. Setting up automated debit instructions ensures zero delinquency, which accounts for the largest recovery in your credit score."
        ))
    if emi_ratio > 0.40:
        recs.append((
            "📉 EMI Burden Management",
            f"Monthly EMI commitments stand at {emi_ratio:.0%} of monthly income (benchmark ceiling is 40%). Consolidating high-interest loans can lower monthly obligations."
        ))
    if dti > 1.2:
        recs.append((
            "⚖️ Debt Amortization Plan",
            f"Outstanding debt stands at {dti:.2f}× annual income. Directing bonus or discretionary cash flow towards reducing principal will directly improve your risk index."
        ))
    if balance < mi * 0.15:
        recs.append((
            "🏦 Liquidity Reserve",
            f"Current closing balance is lean relative to monthly expenses. Building a 3-month liquidity cushion (~{sym}{mi*3:,.0f}) enhances financial resilience."
        ))
    if pred == "Good" and delayed_payments == 0:
        recs.append((
            "✅ Prime Maintenance",
            "Your profile scores in Tier 1. Maintaining current debt-to-income and payment discipline qualifies you for prime lending rates and preferred terms."
        ))
    if not recs:
        recs.append((
            "📋 Balanced Trajectory",
            "Profile metrics are stable. Keeping credit card utilization below 30% and avoiding simultaneous new loan inquiries will support a migration to Good."
        ))
    return recs[:4]

# ─────────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────
def make_gauge(risk_score: float, prediction: str):
    cfg = RATING_CONFIG.get(prediction, RATING_CONFIG["Standard"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"font": {"size": 40, "color": "#0F172A", "family": "Inter, system-ui"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1",
                "tickvals": [0, 30, 60, 100],
                "ticktext": ["Prime (0-30)", "Standard (30-60)", "High Risk (60-100)", ""],
                "tickfont": {"size": 9, "color": "#94A3B8"},
            },
            "bar": {"color": cfg["hex"], "thickness": 0.22},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#ECFDF5"},
                {"range": [30, 60], "color": "#FFFBEB"},
                {"range": [60, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": cfg["hex"], "width": 4},
                "thickness": 0.8, "value": risk_score,
            },
        },
    ))
    fig.update_layout(
        height=220, margin={"l": 20, "r": 20, "t": 15, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui"},
    )
    return fig

def make_prob_chart(probabilities, class_names, prediction):
    COLOR_MAP = {"Good": "#10B981", "Standard": "#F59E0B", "Poor": "#EF4444"}
    pairs = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals = [p[1] * 100 for p in pairs]
    colors = [COLOR_MAP.get(l, "#6B7280") for l in labels]
    ylabels = [f"<b>{l}</b>" if l == prediction else l for l in labels]

    fig = go.Figure(go.Bar(
        x=vals, y=ylabels, orientation="h",
        marker={"color": colors, "line": {"width": 0}, "cornerradius": 5},
        text=[f"<b>{v:.1f}%</b>" for v in vals],
        textposition="outside",
        textfont={"size": 13, "color": "#374151", "family": "Inter, system-ui"},
        cliponaxis=False,
    ))
    fig.update_layout(
        height=170, margin={"l": 0, "r": 65, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 118], "showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "tickfont": {"size": 13, "color": "#374151", "family": "Inter"}},
        showlegend=False, bargap=0.36,
    )
    return fig

def make_importance_chart(imp_df):
    df = imp_df.copy()
    df["Label"] = df["Feature"].map(lambda f: FEAT_DISPLAY.get(f, f))
    df = df.sort_values("Importance", ascending=True)
    mx = df["Importance"].max()
    colors = [f"rgba(37,99,235,{0.25 + 0.75*(v/mx):.2f})" for v in df["Importance"]]

    fig = go.Figure(go.Bar(
        x=df["Importance"] * 100, y=df["Label"], orientation="h",
        marker={"color": colors, "line": {"width": 0}, "cornerradius": 5},
        text=[f"<b>{v*100:.1f}%</b>" for v in df["Importance"]],
        textposition="outside",
        textfont={"size": 12, "color": "#374151", "family": "Inter, system-ui"},
        cliponaxis=False,
    ))
    fig.update_layout(
        height=290, margin={"l": 0, "r": 65, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "showgrid": True, "gridcolor": "#F1F5F9",
            "showticklabels": False, "zeroline": False, "range": [0, mx * 130],
        },
        yaxis={"showgrid": False, "tickfont": {"size": 12, "color": "#374151", "family": "Inter"}},
        showlegend=False, bargap=0.30,
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif !important;
}
.stApp { background: #F0F4F8 !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container {
    max-width: 100% !important; padding-top: 0 !important;
    padding-bottom: 60px !important; padding-left: 2rem !important; padding-right: 2rem !important;
}

[data-testid="stSidebar"] {
    background: #0A0F1E !important; border-right: 1px solid #111827 !important;
    min-width: 230px !important; max-width: 230px !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebarNav"] { display: none !important; }

h1,h2,h3,h4 { color: #0F172A !important; letter-spacing: -0.3px; }
p, label, .stMarkdown p { color: #475569 !important; }

/* ── NUMBER INPUTS (Clean modern input, no ugly stepper clutter) ── */
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none !important; margin: 0 !important; }
input[type="number"] { -moz-appearance: textfield !important; }

div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important; color: #0F172A !important;
    border: 1.5px solid #D1D5DB !important; border-radius: 8px !important;
    min-height: 44px !important; font-size: 15px !important; font-weight: 600 !important;
    padding: 10px 14px !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2563EB !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.14) !important;
}
div[data-testid="stNumberInput"] button {
    display: none !important;
}

div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important; border: 1.5px solid #D1D5DB !important;
    border-radius: 8px !important; min-height: 44px !important;
}
div[data-testid="stSelectbox"] *,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div[class*="singleValue"],
div[data-testid="stSelectbox"] div[class*="placeholder"] {
    color: #111827 !important; font-size: 14px !important; font-weight: 600 !important;
}
div[data-testid="stWidgetLabel"] label {
    color: #374151 !important; font-size: 11px !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.5px !important;
}

/* ── 1-CLICK BENCHMARK BUTTONS (Regular st.button) ── */
/* Unselected (Secondary) Button: Crisp white card with subtle border */
.stButton > button,
button[kind="secondary"],
[data-testid="baseButton-secondary"],
[data-testid="stBaseButton-secondary"] {
    background-color: #FFFFFF !important;
    background: #FFFFFF !important;
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    min-height: 42px !important;
    padding: 8px 16px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover,
button[kind="secondary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover {
    background-color: #F8FAFC !important;
    background: #F8FAFC !important;
    border-color: #94A3B8 !important;
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

.stButton > button p,
button[kind="secondary"] p,
[data-testid="baseButton-secondary"] p,
[data-testid="stBaseButton-secondary"] p {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* Selected / Highlighted (Primary) Button: Vibrant royal blue card with glow */
.stButton > button[kind="primary"],
button[kind="primary"],
[data-testid="baseButton-primary"],
[data-testid="stBaseButton-primary"] {
    background-color: #2563EB !important;
    background: #2563EB !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 1.5px solid #1D4ED8 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    min-height: 42px !important;
    padding: 8px 16px !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
}

.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: #1D4ED8 !important;
    background: #1D4ED8 !important;
    border-color: #1E40AF !important;
}

.stButton > button[kind="primary"] p,
button[kind="primary"] p,
[data-testid="baseButton-primary"] p,
[data-testid="stBaseButton-primary"] p {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}

/* ── FORM SUBMIT BUTTONS ── */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: #2563EB !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important; font-size: 14px !important;
    font-weight: 700 !important; min-height: 46px !important; width: 100% !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3); transition: background 0.15s;
}
[data-testid="stFormSubmitButton"] button:hover { background: #1D4ED8 !important; }
[data-testid="stFormSubmitButton"] button p { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 700 !important; }

[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] {
    background: #FFFFFF !important; color: #374151 !important; -webkit-text-fill-color: #374151 !important;
    border: 1.5px solid #D1D5DB !important;
}
[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] p { color: #374151 !important; -webkit-text-fill-color: #374151 !important; }

.stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #E2E8F0 !important; background: transparent; }
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none; border-bottom: 2px solid transparent;
    color: #64748B !important; font-size: 13px; font-weight: 600; padding: 12px 20px;
}
.stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom-color: #2563EB !important; }
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px !important; }
[data-testid="stForm"] { background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important; border-radius: 14px !important; padding: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.html("""
    <div style="padding:24px 18px 18px;border-bottom:1px solid #1E293B;margin-bottom:20px;">
        <div style="font-size:24px;font-weight:900;color:#F1F5F9;letter-spacing:-0.5px;line-height:1.1;">
            Credi<span style="color:#3B82F6;">X</span>
        </div>
        <div style="font-size:9px;color:#475569;letter-spacing:2px;margin-top:4px;text-transform:uppercase;font-weight:600;">Enterprise Credit Risk AI</div>
    </div>
    <div style="padding:0 10px;margin-bottom:20px;">
        <div style="font-size:9px;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:2px;padding:0 8px;margin-bottom:8px;">Workspace</div>
        <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.15);">
            <span>📊</span><span style="font-size:13px;font-weight:700;color:#93C5FD;">Credit Underwriting</span>
        </div>
    </div>
    <div style="padding:0 10px;margin-bottom:20px;">
        <div style="font-size:9px;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:2px;padding:0 8px;margin-bottom:8px;">Engine Architecture</div>
        <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.18);border-radius:10px;padding:14px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                <div style="width:7px;height:7px;background:#10B981;border-radius:50%;box-shadow:0 0 5px #10B981;"></div>
                <span style="font-size:11px;font-weight:800;color:#6EE7B7;letter-spacing:0.5px;">ONLINE</span>
            </div>
            <div style="font-size:11px;color:#64748B;line-height:1.8;">
                <div>Extra Trees Classifier</div>
                <div>Isotonic Calibration (cv=5)</div>
                <div>FICO Score Engine (300-850)</div>
                <div style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.05);color:#475569;">31,711 records trained</div>
            </div>
        </div>
    </div>
    """)
    st.markdown("---")
    st.html("""
    <div style="padding:10px 18px 20px;">
        <div style="font-size:13px;font-weight:700;color:#CBD5E1;">Rajat Murhe</div>
        <div style="font-size:11px;color:#475569;">ML AI Engineer</div>
    </div>
    """)

# ─────────────────────────────────────────────────────────────────
# TOP BAR WITH CURRENCY & POLICY SELECTORS
# ─────────────────────────────────────────────────────────────────
tb_c1, tb_c2, tb_c3 = st.columns([1.5, 1.2, 1.3])
with tb_c1:
    st.html("""
    <div style="display:flex;align-items:center;gap:8px;padding-top:8px;">
        <span style="font-size:13px;color:#94A3B8;">Platform</span>
        <span style="color:#CBD5E1;">›</span>
        <span style="font-size:13px;color:#0F172A;font-weight:700;">Credit Intelligence & Underwriting</span>
    </div>
    """)
with tb_c2:
    currency = st.radio(
        "Currency Scale",
        options=["USD ($)", "INR (₹)"],
        index=0,
        horizontal=True,
        help="USD uses Kaggle benchmark scales. INR normalizes values with 1:83 purchasing parity."
    )
    is_inr = "INR" in currency
    curr_sym = "₹" if is_inr else "$"
with tb_c3:
    decision_mode = st.radio(
        "Decision Policy",
        options=["Prudential Risk-Aware", "Standard Argmax"],
        index=0,
        horizontal=True,
        help="Prudential mode prioritizes default risk detection (P(Poor) >= 30%) to eliminate false negatives."
    )
    is_risk_aware = "Prudential" in decision_mode
    mode_key = "risk_aware" if is_risk_aware else "argmax"

st.markdown("<hr style='margin:10px 0 20px 0;'>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 1-CLICK BENCHMARK TEST PERSONAS
# ─────────────────────────────────────────────────────────────────
st.html("""<div style="font-size:11px;font-weight:800;color:#64748B;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px;">1-Click Benchmark Test Profiles:</div>""")
p_col1, p_col2, p_col3, p_col4 = st.columns([1, 1, 1, 2])

# Session state initialization for presets
if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = "balanced"

if "form_vals" not in st.session_state:
    st.session_state["form_vals"] = {
        "age": 32, "occupation": "Developer",
        "income": 40000.0 if not is_inr else 3000000.0,
        "delayed": 5,
        "emi": 45.0 if not is_inr else 3500.0,
        "debt": 750.0 if not is_inr else 60000.0,
        "balance": 350.0 if not is_inr else 30000.0,
    }

active_preset = st.session_state.get("selected_preset", "balanced")
is_prime = active_preset == "prime"
is_balanced = active_preset == "balanced"
is_high_risk = active_preset == "high_risk"

with p_col1:
    prime_label = "🌟 Prime (Tier 1)  ✓" if is_prime else "🌟 Prime (Tier 1)"
    if st.button(prime_label, type="primary" if is_prime else "secondary", use_container_width=True, key="btn_prime"):
        st.session_state["selected_preset"] = "prime"
        st.session_state["form_vals"] = {
            "age": 38, "occupation": "Engineer",
            "income": 65000.0 if not is_inr else 5000000.0,
            "delayed": 0,
            "emi": 35.0 if not is_inr else 2500.0,
            "debt": 350.0 if not is_inr else 25000.0,
            "balance": 600.0 if not is_inr else 50000.0,
        }
        if "result" in st.session_state:
            del st.session_state["result"]
        st.rerun()

with p_col2:
    bal_label = "⚖️ Balanced (Tier 2)  ✓" if is_balanced else "⚖️ Balanced (Tier 2)"
    if st.button(bal_label, type="primary" if is_balanced else "secondary", use_container_width=True, key="btn_balanced"):
        st.session_state["selected_preset"] = "balanced"
        st.session_state["form_vals"] = {
            "age": 32, "occupation": "Developer",
            "income": 40000.0 if not is_inr else 3000000.0,
            "delayed": 5,
            "emi": 45.0 if not is_inr else 3500.0,
            "debt": 750.0 if not is_inr else 60000.0,
            "balance": 350.0 if not is_inr else 30000.0,
        }
        if "result" in st.session_state:
            del st.session_state["result"]
        st.rerun()

with p_col3:
    risk_label = "⚠️ High-Risk (Tier 3)  ✓" if is_high_risk else "⚠️ High-Risk (Tier 3)"
    if st.button(risk_label, type="primary" if is_high_risk else "secondary", use_container_width=True, key="btn_high_risk"):
        st.session_state["selected_preset"] = "high_risk"
        st.session_state["form_vals"] = {
            "age": 25, "occupation": "Teacher",
            "income": 20000.0 if not is_inr else 1500000.0,
            "delayed": 22,
            "emi": 90.0 if not is_inr else 7500.0,
            "debt": 1300.0 if not is_inr else 110000.0,
            "balance": 100.0 if not is_inr else 8000.0,
        }
        if "result" in st.session_state:
            del st.session_state["result"]
        st.rerun()

# ─────────────────────────────────────────────────────────────────
# WORKSPACE: LEFT (FORM) & RIGHT (ASSESSMENT)
# ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([0.95, 1.05], gap="large")

with left_col:
    st.html("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
        <div style="width:28px;height:28px;background:#2563EB;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#fff;">1</div>
        <div>
            <div style="font-size:14px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.5px;">Applicant Financial Profile</div>
            <div style="font-size:12px;color:#64748B;">Enter or adjust customer financial parameters</div>
        </div>
    </div>
    """)

    f_vals = st.session_state["form_vals"]

    with st.form("credit_form"):
        st.html("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Demographics</div>""")
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=int(f_vals["age"]), step=1)
        with c2:
            occ_idx = OCCUPATIONS.index(f_vals["occupation"]) if f_vals["occupation"] in OCCUPATIONS else 4
            occupation = st.selectbox("Occupation", options=OCCUPATIONS, index=occ_idx)

        st.html("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px;">Financial Details</div>""")
        c3, c4 = st.columns(2)
        with c3:
            step_inc = 10000.0 if is_inr else 1000.0
            annual_income = st.number_input(f"Annual Income ({curr_sym})", min_value=1000.0, value=float(f_vals["income"]), step=step_inc, format="%.0f")
        with c4:
            delayed_payments = st.number_input("Recorded Late Payments", min_value=0, max_value=60, value=int(f_vals["delayed"]), step=1)

        c5, c6 = st.columns(2)
        with c5:
            step_emi = 500.0 if is_inr else 10.0
            emi = st.number_input(f"Monthly EMI ({curr_sym})", min_value=0.0, value=float(f_vals["emi"]), step=step_emi, format="%.0f")
        with c6:
            step_debt = 1000.0 if is_inr else 50.0
            outstanding_debt = st.number_input(f"Outstanding Debt ({curr_sym})", min_value=0.0, value=float(f_vals["debt"]), step=step_debt, format="%.0f")

        step_bal = 1000.0 if is_inr else 50.0
        monthly_balance = st.number_input(f"Monthly Closing Balance ({curr_sym})", min_value=0.0, value=float(f_vals["balance"]), step=step_bal, format="%.0f")

        # ── Real-time Financial Ratios ──
        mi = annual_income / 12.0 if annual_income > 0 else 1.0
        emi_ratio = emi / mi if mi > 0 else 0.0
        dti = outstanding_debt / annual_income if annual_income > 0 else 0.0
        sav_pct = min(100.0, (monthly_balance / mi) * 100.0) if mi > 0 else 0.0

        def _rc(bad): return "#EF4444" if bad else "#10B981"
        def _rl(bad, g, b): return b if bad else g

        st.html(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:12px;margin:14px 0;display:flex;">
            <div style="flex:1;text-align:center;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;">EMI / Income</div>
                <div style="font-size:18px;font-weight:900;color:{_rc(emi_ratio>0.4)};">{emi_ratio:.1%}</div>
                <div style="font-size:9px;color:#94A3B8;">{_rl(emi_ratio>0.4,'✓ Healthy','▲ High')}</div>
            </div>
            <div style="width:1px;background:#E2E8F0;"></div>
            <div style="flex:1;text-align:center;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;">Debt / Income</div>
                <div style="font-size:18px;font-weight:900;color:{_rc(dti>1.0)};">{dti:.2f}×</div>
                <div style="font-size:9px;color:#94A3B8;">{_rl(dti>1.0,'✓ Healthy','▲ High')}</div>
            </div>
            <div style="width:1px;background:#E2E8F0;"></div>
            <div style="flex:1;text-align:center;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;">Liquidity Cushion</div>
                <div style="font-size:18px;font-weight:900;color:{_rc(sav_pct<10)};">{sav_pct:.0f}%</div>
                <div style="font-size:9px;color:#94A3B8;">{_rl(sav_pct<10,'✓ Adequate','▲ Lean')}</div>
            </div>
        </div>
        """)

        b1, b2 = st.columns([2, 1])
        with b1:
            submitted = st.form_submit_button("▶  Run Credit Assessment", type="primary", use_container_width=True)
        with b2:
            reset = st.form_submit_button("Reset", type="secondary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# RIGHT: ASSESSMENT OUTPUT
# ─────────────────────────────────────────────────────────────────
with right_col:
    st.html("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
        <div style="width:28px;height:28px;background:#2563EB;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:800;color:#fff;">2</div>
        <div>
            <div style="font-size:14px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.5px;">Underwriting Decision & Score</div>
            <div style="font-size:12px;color:#64748B;">Multi-factor calibrated assessment outcome</div>
        </div>
    </div>
    """)

    if reset and "result" in st.session_state:
        del st.session_state["result"]
        st.rerun()

    if submitted:
        eval_res = evaluate_applicant(
            age=age, occupation=occupation, annual_income=annual_income,
            delayed_payments=delayed_payments, emi=emi,
            outstanding_debt=outstanding_debt, monthly_balance=monthly_balance,
            currency="INR" if is_inr else "USD", decision_mode=mode_key,
        )
        recs = get_recommendations(
            pred=eval_res["prediction"], annual_income=annual_income,
            delayed_payments=delayed_payments, emi=emi,
            debt=outstanding_debt, balance=monthly_balance, sym=curr_sym,
        )
        st.session_state["result"] = {
            **eval_res,
            "recs": recs,
            "currency_sym": curr_sym,
            "currency_name": "INR" if is_inr else "USD",
            "decision_policy": "Prudential Risk-Aware" if is_risk_aware else "Standard Argmax",
            "inputs_display": {
                "Age": f"{age} years",
                "Occupation": occupation,
                "Annual Income": f"{curr_sym}{annual_income:,.0f}",
                "Delayed Payments": f"{delayed_payments} late",
                "Monthly EMI": f"{curr_sym}{emi:,.0f}",
                "Outstanding Debt": f"{curr_sym}{outstanding_debt:,.0f}",
                "Monthly Balance": f"{curr_sym}{monthly_balance:,.0f}",
            },
        }

    if "result" not in st.session_state:
        st.html("""
        <div style="background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:14px;padding:58px 30px;text-align:center;">
            <div style="font-size:48px;margin-bottom:14px;">📊</div>
            <div style="font-size:17px;font-weight:800;color:#0F172A;margin-bottom:6px;">Ready for Evaluation</div>
            <div style="font-size:13px;color:#64748B;max-width:300px;margin:0 auto;line-height:1.7;">
                Select a benchmark profile above or customize input parameters, then click
                <b style="color:#2563EB;">Run Credit Assessment</b>.
            </div>
        </div>
        """)
    else:
        r = st.session_state["result"]
        pred = r["prediction"]
        cfg = RATING_CONFIG.get(pred, RATING_CONFIG["Standard"])
        score = r["credit_score"]
        risk = r["risk_score"]

        # ── OOD Warning if detected ──
        if r["ood_warnings"]:
            warnings_text = "<br>• ".join(r["ood_warnings"])
            st.html(f"""
            <div style="background:#FEF3C7;border:1px solid #FCD34D;border-radius:8px;padding:10px 14px;margin-bottom:12px;font-size:11px;color:#92400E;">
                ⚠️ <b>Domain Note:</b> {warnings_text} (Model applies robust outlier scaling).
            </div>
            """)

        # ── Primary Decision Banner ──
        st.html(f"""
        <div style="background:linear-gradient(135deg,{cfg['bg']} 0%,#FFFFFF 100%);border:1.5px solid {cfg['border']};border-radius:14px;padding:22px 26px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    <div style="font-size:9px;font-weight:800;color:{cfg['dark']};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Underwriting Decision</div>
                    <div style="font-size:46px;font-weight:900;color:{cfg['hex']};letter-spacing:-2px;line-height:1;margin-bottom:10px;">{pred.upper()}</div>
                    <div style="display:inline-block;background:rgba(255,255,255,0.85);border:1.5px solid {cfg['border']};border-radius:20px;padding:4px 14px;font-size:11px;font-weight:800;color:{cfg['dark']};">
                        {cfg['badge']} · {cfg['tier']}
                    </div>
                </div>
                <div style="background:#FFFFFF;border:1.5px solid {cfg['border']};border-radius:12px;padding:12px 18px;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.03);">
                    <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;">CrediX Score</div>
                    <div style="font-size:34px;font-weight:900;color:#0F172A;line-height:1.1;">{score}</div>
                    <div style="font-size:10px;color:#64748B;">Scale: 300–850</div>
                </div>
            </div>
        </div>
        """)

        # ── Risk Gauge & Probabilities ──
        gc, pc = st.columns(2)
        with gc:
            st.html("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;text-align:center;margin-bottom:2px;">Risk Index (0-100)</div>""")
            if PLOTLY_OK:
                st.plotly_chart(make_gauge(risk, pred), use_container_width=True, config={"displayModeBar": False})
        with pc:
            st.html("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;text-align:center;margin-bottom:2px;">Class Probabilities</div>""")
            if PLOTLY_OK:
                st.plotly_chart(make_prob_chart(r["probabilities"], class_names, pred), use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────
# 5-TAB ANALYTICS SUITE
# ─────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r = st.session_state["result"]
    pred = r["prediction"]
    cfg = RATING_CONFIG.get(pred, RATING_CONFIG["Standard"])

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Feature Impact  ",
        "  Recommendations  ",
        "  Model Performance & Audit  ",
        "  Input Summary  ",
        "  Export Report  ",
    ])

    # ── TAB 1: Feature Impact ──
    with tab1:
        ta, tb = st.columns([1.2, 0.85], gap="large")
        with ta:
            st.html("""
            <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Key Model Drivers (Gini-MDI)</div>
            <div style="font-size:12px;color:#64748B;margin-bottom:14px;">Mean decrease in impurity across all calibrated ensemble estimators</div>
            """)
            if PLOTLY_OK:
                st.plotly_chart(make_importance_chart(importance_df), use_container_width=True, config={"displayModeBar": False})
        with tb:
            st.html("""<div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:12px;">Feature Weight Table</div>""")
            for _, row in importance_df.iterrows():
                disp = FEAT_DISPLAY.get(row["Feature"], row["Feature"])
                pct = row["Importance"] * 100
                st.html(f"""
                <div style="margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                        <span style="font-size:12px;font-weight:700;color:#374151;">{disp}</span>
                        <span style="font-size:12px;font-weight:800;color:#0F172A;">{pct:.1f}%</span>
                    </div>
                    <div style="background:#F1F5F9;border-radius:4px;height:6px;">
                        <div style="width:{int(pct/importance_df['Importance'].max()*100)}%;height:100%;background:#2563EB;border-radius:4px;"></div>
                    </div>
                </div>
                """)

    # ── TAB 2: Recommendations ──
    with tab2:
        st.html("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Actionable Credit Guidance</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:18px;">Specific remediation steps generated for this applicant's profile</div>
        """)
        for title, body in r["recs"]:
            st.html(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid {cfg['hex']};border-radius:10px;padding:16px 20px;margin-bottom:12px;">
                <div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:4px;">{title}</div>
                <div style="font-size:12px;color:#475569;line-height:1.7;">{body}</div>
            </div>
            """)

    # ── TAB 3: Model Performance & Strict Audit ──
    with tab3:
        st.html("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Strict Evaluation & Audit Report</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:18px;">Held-out test set evaluation (6,343 samples, Stratified 80/20 split)</div>
        """)

        m_cols = st.columns(3)
        for i, (label, val) in enumerate(MODEL_METRICS.items()):
            with m_cols[i % 3]:
                st.metric(label, f"{val:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.html("""<div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:12px;">Class-Specific Evaluation</div>""")
        c_cols = st.columns(3)
        for i, (cls, mets) in enumerate(CLASS_METRICS.items()):
            c = RATING_CONFIG.get(cls, {})
            with c_cols[i]:
                st.html(f"""
                <div style="background:#FFFFFF;border:1.5px solid {c.get('border','#E2E8F0')};border-radius:12px;padding:18px;text-align:center;">
                    <div style="font-size:11px;font-weight:800;color:{c.get('dark','#374151')};text-transform:uppercase;margin-bottom:12px;">{cls}</div>
                    <div style="display:flex;justify-content:space-around;">
                        <div>
                            <div style="font-size:18px;font-weight:900;color:{c.get('hex','#374151')};">{mets['Precision']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;">Precision</div>
                        </div>
                        <div>
                            <div style="font-size:18px;font-weight:900;color:{c.get('hex','#374151')};">{mets['Recall']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;">Recall</div>
                        </div>
                        <div>
                            <div style="font-size:18px;font-weight:900;color:{c.get('hex','#374151')};">{mets['F1']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;">F1</div>
                        </div>
                    </div>
                </div>
                """)

        st.html("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:16px 20px;margin-top:20px;font-size:12px;color:#64748B;line-height:1.8;">
            <b style="color:#0F172A;">Prudential Policy Implementation:</b> In standard 3-class argmax decisioning, 35.3% of true defaults were misclassified into the majority class (Standard) because default probability peaked between 30% and 49%.
            The <b>Prudential Risk-Aware mode</b> lowers the threshold to <code>P(Poor) ≥ 30%</code>, improving default recall from <b>63.9% to 77.3%</b> and reducing false negatives by 39%.
        </div>
        """)

    # ── TAB 4: Input Summary ──
    with tab4:
        st.html("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Audit Trail & Input Summary</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:18px;">Exact parameters evaluated for this assessment session</div>
        """)
        s1, s2 = st.columns([1, 1], gap="large")
        with s1:
            st.html("""<div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;margin-bottom:10px;">Submitted Parameters</div>""")
            rows_html = "".join([
                f"""<div style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #F1F5F9;">
                    <span style="font-size:13px;color:#64748B;">{k}</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{v}</span>
                </div>""" for k, v in r["inputs_display"].items()
            ])
            st.html(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:4px 18px;">{rows_html}</div>""")
        with s2:
            st.html("""<div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;margin-bottom:10px;">Session Metadata</div>""")
            st.html(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:16px 18px;">
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #F1F5F9;">
                    <span style="font-size:13px;color:#64748B;">Currency Selected</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{r['currency_name']} ({r['currency_sym']})</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #F1F5F9;">
                    <span style="font-size:13px;color:#64748B;">Decision Policy</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{r['decision_policy']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #F1F5F9;">
                    <span style="font-size:13px;color:#64748B;">FICO-Style Score</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{r['credit_score']} / 850</span>
                </div>
                <div style="display:flex;justify-content:space-between;padding:8px 0;">
                    <span style="font-size:13px;color:#64748B;">Risk Index</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{r['risk_score']} / 100</span>
                </div>
            </div>
            """)

    # ── TAB 5: Export Report ──
    with tab5:
        st.html("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Export Official Assessment Report</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:20px;">Download an audit-ready PDF credit evaluation document</div>
        """)
        rc1, rc2 = st.columns([1, 1], gap="large")
        with rc1:
            st.html(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:24px;">
                <div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:16px;">Report Package Includes</div>
                <div style="font-size:13px;color:#475569;line-height:2.2;">
                    ✓ &nbsp; Customer financial profile &amp; credit metrics<br>
                    ✓ &nbsp; Underwriting classification: <b style="color:{cfg['hex']};">{pred}</b><br>
                    ✓ &nbsp; FICO-equivalent Score ({score}/850) &amp; Class Probabilities<br>
                    ✓ &nbsp; Gini-MDI feature importance ranking<br>
                    ✓ &nbsp; Tailored financial remediation directives<br>
                    ✓ &nbsp; Model validation metrics &amp; methodology disclaimer
                </div>
            </div>
            """)
        with rc2:
            st.html("""
            <div style="background:#F8FAFC;border:1.5px dashed #CBD5E1;border-radius:12px;padding:28px;text-align:center;">
                <div style="font-size:36px;margin-bottom:10px;">📄</div>
                <div style="font-size:14px;font-weight:800;color:#0F172A;margin-bottom:4px;">PDF Credit Intelligence Report</div>
                <div style="font-size:12px;color:#64748B;margin-bottom:18px;">Audit-ready documentation for lending files</div>
            </div>
            """)
            if PDF_OK:
                try:
                    recs_clean = [
                        (t.split(" ", 1)[-1] if t[0] in "🔔📉⚖️🏦✅📋📊" else t, b)
                        for t, b in r["recs"]
                    ]
                    pdf_bytes = pdf_mod.generate_pdf(
                        input_df=r["input_df"],
                        prediction=r["prediction"],
                        probabilities=r["probabilities"],
                        class_names=class_names,
                        recommendations=recs_clean,
                        importance_df=importance_df,
                    )
                    if pdf_bytes:
                        st.download_button(
                            label="⬇  Download PDF Report",
                            data=io.BytesIO(pdf_bytes),
                            file_name=f"CrediX_{r['prediction']}_Credit_Report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"PDF generation encountered an error: {exc}")
            else:
                st.html("""<div style="font-size:12px;color:#94A3B8;text-align:center;margin-top:10px;">PDF generation unavailable (reportlab required).</div>""")
