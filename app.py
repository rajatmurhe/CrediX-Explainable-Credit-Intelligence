"""
CrediX — Explainable Credit Intelligence  v2.0
Enterprise Credit Risk Assessment Platform
Complete redesign: Plotly gauges · risk score · tabbed analytics · live ratios
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
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CrediX | Credit Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "credit_data.csv"

EXPECTED_FEATURES = [
    "Age", "Occupation", "Annual_Income",
    "Num_of_Delayed_Payment", "Total_EMI_per_month",
    "Outstanding_Debt", "Monthly_Balance",
]

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media_Manager", "Musician", "Scientist", "Teacher", "Writer",
]

RATING = {
    "Good":     {"hex": "#10B981", "bg": "#ECFDF5", "border": "#A7F3D0", "dark": "#065F46", "muted": "#D1FAE5"},
    "Standard": {"hex": "#F59E0B", "bg": "#FFFBEB", "border": "#FDE68A", "dark": "#78350F", "muted": "#FEF3C7"},
    "Poor":     {"hex": "#EF4444", "bg": "#FEF2F2", "border": "#FECACA", "dark": "#991B1B", "muted": "#FEE2E2"},
}

FEAT_DISPLAY = {
    "Outstanding_Debt": "Outstanding Debt", "Age": "Age",
    "Total_EMI_per_month": "Total Monthly EMI",
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
# LOAD ARTIFACTS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        return joblib.load(BASE_DIR / "trained_credit_model.joblib")
    except Exception as e:
        st.error(f"CrediX model could not be loaded.\n\n{e}")
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

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv(DATA_PATH)
    except Exception:
        return None

model       = load_model()
encoder     = load_encoder()
feat_order  = load_features()
dataset     = load_dataset()
class_names = list(encoder.classes_) if encoder else ["Good", "Poor", "Standard"]

# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
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
        for attr in ("estimator", "base_estimator"):
            inner = getattr(est, attr, None)
            if inner:
                r = _extract(inner)
                if r is not None:
                    return r
        return None

    try:
        raw  = _extract(model)
        prep = None
        if hasattr(model, "named_steps"):
            for k in ("preprocessor", "transformer"):
                if k in model.named_steps:
                    prep = model.named_steps[k]; break
        if raw is None or prep is None:
            raise ValueError("extraction failed")

        col_names = prep.get_feature_names_out()
        agg: dict = {}
        for feat, imp in zip(col_names, raw):
            key  = feat.split("__")[-1] if "__" in feat else feat
            base = "Occupation" if key.startswith("Occupation") else key
            agg[base] = agg.get(base, 0.0) + float(imp)

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
# HELPERS
# ─────────────────────────────────────────────────────────────────
def H(s: str):
    """Render raw HTML block."""
    st.html(s)

def compute_risk_score(probabilities, class_names):
    weights = {"Good": 14.0, "Standard": 52.0, "Poor": 88.0}
    return round(float(sum(probabilities[i] * weights.get(c, 50.0)
                           for i, c in enumerate(class_names))), 1)

def build_input_df(age, occupation, annual_income, delayed_payments, emi, outstanding_debt, monthly_balance):
    raw = {
        "Age": age, "Occupation": occupation,
        "Annual_Income": float(annual_income),
        "Num_of_Delayed_Payment": int(delayed_payments),
        "Total_EMI_per_month": float(emi),
        "Outstanding_Debt": float(outstanding_debt),
        "Monthly_Balance": float(monthly_balance),
    }
    return pd.DataFrame([{f: raw[f] for f in feat_order}])

def run_prediction(input_df):
    pred_enc      = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    label         = encoder.inverse_transform(pred_enc)[0] if encoder else str(pred_enc[0])
    return label, probabilities

def get_recommendations(pred, annual_income, delayed_payments, emi, outstanding_debt, monthly_balance):
    recs = []
    mi        = annual_income / 12 if annual_income > 0 else 1
    emi_ratio = emi / mi         if mi > 0           else 0
    dti       = outstanding_debt / annual_income if annual_income > 0 else 0

    if delayed_payments > 3:
        recs.append(("🔔 Payment History",
            f"You have {int(delayed_payments)} recorded late payments. Set up autopay or calendar reminders — consistent on-time payments are the single biggest driver of credit improvement."))
    if emi_ratio > 0.40:
        recs.append(("📉 EMI Burden",
            f"Annual EMI commitment is {emi_ratio:.0%} of monthly income — above the recommended 40% ceiling. Consider consolidating loans or prepaying high-interest obligations."))
    if dti > 1.5:
        recs.append(("⚖️ Debt Load",
            f"Outstanding debt stands at {dti:.1f}× annual income. A structured debt-reduction plan focused on high-APR accounts can meaningfully lower your risk score within 12 months."))
    if monthly_balance < mi * 0.10:
        recs.append(("🏦 Liquidity Reserve",
            f"Monthly balance is low relative to income. Building a 3-month emergency fund (~₹{annual_income/4:,.0f}) improves both financial resilience and perceived creditworthiness."))
    if pred == "Good" and delayed_payments == 0:
        recs.append(("✅ Sustain Excellence",
            "Your profile scores well across all dimensions. Continue disciplined on-time payments and avoid taking on debt disproportionate to income to maintain your Good rating."))
    if pred == "Standard" and not recs:
        recs.append(("📋 Incremental Improvement",
            "Your profile sits at Standard. Reducing outstanding debt by 20% and eliminating any new delayed payments over the next 6 months could shift you to Good."))
    if not recs:
        recs.append(("📊 Ongoing Monitoring",
            "No specific risk flags detected. Review your financial ratios quarterly and maintain current payment discipline."))
    return recs[:4]

# ─────────────────────────────────────────────────────────────────
# PLOTLY CHARTS
# ─────────────────────────────────────────────────────────────────
def make_gauge(risk_score: float, prediction: str):
    cfg = RATING.get(prediction, RATING["Standard"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        number={"font": {"size": 42, "color": "#0F172A", "family": "Inter, system-ui"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1",
                "tickvals": [0, 35, 65, 100],
                "ticktext": ["Low", "Medium", "High", "Critical"],
                "tickfont": {"size": 9, "color": "#94A3B8"},
            },
            "bar":  {"color": cfg["hex"], "thickness": 0.22},
            "bgcolor": "white", "borderwidth": 0,
            "steps": [
                {"range": [0,  35], "color": "#ECFDF5"},
                {"range": [35, 65], "color": "#FFFBEB"},
                {"range": [65, 100], "color": "#FEF2F2"},
            ],
            "threshold": {
                "line": {"color": cfg["hex"], "width": 4},
                "thickness": 0.8, "value": risk_score,
            },
        },
    ))
    fig.update_layout(
        height=240, margin={"l": 20, "r": 20, "t": 15, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui"},
    )
    return fig

def make_prob_chart(probabilities, class_names, prediction):
    COLOR = {"Good": "#10B981", "Standard": "#F59E0B", "Poor": "#EF4444"}
    pairs  = sorted(zip(class_names, probabilities), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in pairs]
    vals   = [p[1] * 100  for p in pairs]
    colors = [COLOR.get(l, "#6B7280") for l in labels]
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
        height=180, margin={"l": 0, "r": 70, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={"range": [0, 118], "showgrid": False, "showticklabels": False, "zeroline": False},
        yaxis={"showgrid": False, "tickfont": {"size": 13, "color": "#374151", "family": "Inter"}},
        showlegend=False, bargap=0.38,
    )
    return fig

def make_importance_chart(imp_df):
    df  = imp_df.copy()
    df["Label"] = df["Feature"].map(lambda f: FEAT_DISPLAY.get(f, f))
    df  = df.sort_values("Importance", ascending=True)
    mx  = df["Importance"].max()
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
        height=300, margin={"l": 0, "r": 70, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis={
            "showgrid": True, "gridcolor": "#F1F5F9",
            "showticklabels": False, "zeroline": False, "range": [0, mx * 132],
        },
        yaxis={"showgrid": False, "tickfont": {"size": 12, "color": "#374151", "family": "Inter"}},
        showlegend=False, bargap=0.32,
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# CSS
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

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #0A0F1E !important; border-right: 1px solid #111827 !important;
    min-width: 230px !important; max-width: 230px !important;
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebarNav"] { display: none !important; }

h1,h2,h3,h4 { color: #0F172A !important; letter-spacing: -0.3px; }
p, label, .stMarkdown p { color: #475569 !important; }

/* ── NUMBER INPUTS ── */
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none !important; margin: 0 !important; }
input[type="number"] { -moz-appearance: textfield !important; }

div[data-testid="stNumberInput"] input {
    background: #FFFFFF !important; color: #0F172A !important;
    border: 1.5px solid #D1D5DB !important; border-radius: 8px !important;
    min-height: 46px !important; font-size: 15px !important; font-weight: 600 !important;
    padding: 10px 14px !important; box-shadow: inset 0 1px 2px rgba(0,0,0,0.03) !important;
    transition: border-color 0.15s, box-shadow 0.15s;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: #2563EB !important; outline: none !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.14) !important;
}
div[data-testid="stNumberInput"] button {
    background: #F9FAFB !important; color: #6B7280 !important;
    border: 1.5px solid #E5E7EB !important; border-radius: 6px !important;
    width: 30px !important; min-height: 30px !important; max-height: 30px !important;
    font-size: 14px !important; font-weight: 700 !important;
    padding: 0 !important; margin: 2px !important; box-shadow: none !important; cursor: pointer !important;
}
div[data-testid="stNumberInput"] button:hover { background: #E5E7EB !important; color: #111827 !important; border-color: #D1D5DB !important; }
div[data-testid="stNumberInput"] button p { color: #6B7280 !important; font-weight: 800 !important; }
div[data-testid="stNumberInput"] button:hover p { color: #111827 !important; }

/* ── SELECTBOX ── */
div[data-testid="stSelectbox"] > div > div,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #FFFFFF !important; border: 1.5px solid #D1D5DB !important;
    border-radius: 8px !important; min-height: 46px !important;
}
div[data-testid="stSelectbox"] > div > div:focus-within,
div[data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
    border-color: #2563EB !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.14) !important;
}
div[data-testid="stSelectbox"] *,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] div[class*="singleValue"],
div[data-testid="stSelectbox"] div[class*="placeholder"] {
    color: #111827 !important; font-size: 15px !important; font-weight: 600 !important;
}
div[data-testid="stSelectbox"] svg { color: #6B7280 !important; fill: #6B7280 !important; }
[data-baseweb="popover"] li, [data-baseweb="menu"] li { color: #111827 !important; background: #FFFFFF !important; font-size: 14px !important; }
[data-baseweb="popover"] [aria-selected="true"], [data-baseweb="menu"] [aria-selected="true"],
[data-baseweb="popover"] li:hover, [data-baseweb="menu"] li:hover { background: #EFF6FF !important; color: #1D4ED8 !important; }

/* ── LABELS ── */
div[data-testid="stWidgetLabel"] label {
    color: #374151 !important; font-size: 11px !important; font-weight: 800 !important;
    text-transform: uppercase !important; letter-spacing: 0.5px !important;
}

/* ── PRIMARY FORM SUBMIT ── */
[data-testid="stFormSubmitButton"] button,
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
    background: #2563EB !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important; font-size: 14px !important;
    font-weight: 700 !important; min-height: 48px !important; width: 100% !important;
    cursor: pointer !important; opacity: 1 !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.3), 0 1px 2px rgba(0,0,0,0.06);
    transition: background 0.15s, box-shadow 0.15s, transform 0.1s;
}
[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"]:hover {
    background: #1D4ED8 !important; box-shadow: 0 6px 18px rgba(37,99,235,0.4) !important; transform: translateY(-1px);
}
[data-testid="stFormSubmitButton"] button p,
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] p {
    color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 700 !important;
}

/* ── SECONDARY FORM SUBMIT ── */
[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] {
    background: #FFFFFF !important; color: #374151 !important; -webkit-text-fill-color: #374151 !important;
    border: 1.5px solid #D1D5DB !important; box-shadow: none !important;
}
[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"]:hover { background: #F9FAFB !important; transform: none; }
[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] p { color: #374151 !important; -webkit-text-fill-color: #374151 !important; }

/* ── st.button ── */
.stButton > button {
    background: #2563EB !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important; font-size: 14px !important;
    font-weight: 700 !important; min-height: 44px !important; cursor: pointer !important; opacity: 1 !important;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25);
}
.stButton > button p { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 700 !important; }
.stButton > button:hover { background: #1D4ED8 !important; }
.stButton > button:disabled { opacity: 1 !important; }

/* ── DOWNLOAD ── */
[data-testid="stDownloadButton"] button {
    background: #2563EB !important; color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
    min-height: 48px !important; width: 100% !important; cursor: pointer !important;
}
[data-testid="stDownloadButton"] button p { color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 700 !important; }
[data-testid="stDownloadButton"] button:hover { background: #1D4ED8 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 2px solid #E2E8F0 !important; background: transparent;
    margin-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; border: none; border-bottom: 2px solid transparent;
    color: #64748B !important; font-size: 13px; font-weight: 600;
    padding: 12px 22px; margin-bottom: -2px; letter-spacing: 0.1px;
    transition: color 0.15s;
}
.stTabs [aria-selected="true"] { color: #2563EB !important; border-bottom-color: #2563EB !important; background: transparent; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 0 0 0; }

/* ── EXPANDERS ── */
.streamlit-expanderHeader { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 6px !important; color: #0F172A !important; font-weight: 600 !important; }
.streamlit-expanderContent { border: 1px solid #E2E8F0 !important; border-top: none !important; border-radius: 0 0 6px 6px !important; }

/* ── METRICS ── */
div[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 18px !important; }
div[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 11px !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.4px !important; }
div[data-testid="stMetricValue"] { color: #0F172A !important; font-size: 24px !important; font-weight: 800 !important; }

/* ── FORM ── */
[data-testid="stForm"] { background: #FFFFFF !important; border: 1.5px solid #E2E8F0 !important; border-radius: 14px !important; padding: 12px !important; }

hr { border-color: #E2E8F0 !important; margin: 20px 0 !important; }
.stCaption { color: #94A3B8 !important; font-size: 12px !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border: 1.5px solid #E2E8F0 !important; border-radius: 10px !important; background: #FFFFFF !important; padding: 20px !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #F0F4F8; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    H("""
    <div style="padding:26px 20px 20px;border-bottom:1px solid #1E293B;margin-bottom:22px;">
        <div style="font-size:24px;font-weight:900;color:#F1F5F9;letter-spacing:-0.5px;line-height:1.1;">
            Credi<span style="color:#3B82F6;">X</span>
        </div>
        <div style="font-size:9px;color:#475569;letter-spacing:2px;margin-top:4px;text-transform:uppercase;font-weight:600;">Credit Intelligence</div>
    </div>

    <div style="padding:0 12px;margin-bottom:22px;">
        <div style="font-size:9px;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:2px;padding:0 8px;margin-bottom:10px;">Navigation</div>

        <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;background:rgba(59,130,246,0.15);margin-bottom:3px;border:1px solid rgba(59,130,246,0.15);">
            <span style="font-size:14px;">📊</span>
            <span style="font-size:13px;font-weight:700;color:#93C5FD;">Credit Assessment</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;margin-bottom:3px;">
            <span style="font-size:14px;opacity:0.35;">🔍</span>
            <span style="font-size:13px;font-weight:500;color:#475569;">Model Insights</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:8px;margin-bottom:3px;">
            <span style="font-size:14px;opacity:0.35;">ℹ️</span>
            <span style="font-size:13px;font-weight:500;color:#475569;">Model Information</span>
        </div>
    </div>

    <div style="padding:0 12px;margin-bottom:22px;">
        <div style="font-size:9px;font-weight:800;color:#334155;text-transform:uppercase;letter-spacing:2px;padding:0 8px;margin-bottom:10px;">Model Status</div>
        <div style="background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.18);border-radius:10px;padding:14px 16px;">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
                <div style="width:8px;height:8px;background:#10B981;border-radius:50%;box-shadow:0 0 6px rgba(16,185,129,0.6);flex-shrink:0;"></div>
                <span style="font-size:11px;font-weight:800;color:#6EE7B7;letter-spacing:0.5px;">ACTIVE</span>
            </div>
            <div style="font-size:11px;color:#64748B;line-height:1.9;">
                <div>Extra Trees Classifier</div>
                <div>Isotonic Calibration</div>
                <div>3-Class Output</div>
                <div style="margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.05);color:#475569;">31,711 training records</div>
            </div>
        </div>
    </div>
    """)

    st.markdown("---")

    H("""
    <div style="padding:14px 20px 22px;">
        <div style="font-size:13px;font-weight:700;color:#CBD5E1;margin-bottom:2px;">Rajat Murhe</div>
        <div style="font-size:11px;color:#475569;">ML AI Engineer</div>
    </div>
    """)

# ─────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────
H("""
<div style="background:#FFFFFF;border-bottom:1px solid #E2E8F0;padding:12px 24px;display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;margin-left:-2rem;margin-right:-2rem;position:sticky;top:0;z-index:999;box-shadow:0 1px 6px rgba(0,0,0,0.04);">
    <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:13px;color:#94A3B8;">Platform</span>
        <span style="color:#CBD5E1;font-size:13px;">›</span>
        <span style="font-size:13px;color:#0F172A;font-weight:700;">Credit Assessment</span>
    </div>
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:20px;padding:5px 14px;display:flex;align-items:center;gap:7px;">
            <div style="width:7px;height:7px;background:#22C55E;border-radius:50%;box-shadow:0 0 5px rgba(34,197,94,0.5);"></div>
            <span style="font-size:11px;font-weight:800;color:#166534;letter-spacing:0.5px;">EXTRA TREES ACTIVE</span>
        </div>
    </div>
</div>
""")

# ─────────────────────────────────────────────────────────────────
# TWO-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([0.95, 1.05], gap="large")

# ─────────────────────────────────────────────────────────────────
# LEFT — INPUT FORM
# ─────────────────────────────────────────────────────────────────
with left_col:
    H("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
        <div style="width:30px;height:30px;background:#2563EB;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">1</div>
        <div>
            <div style="font-size:14px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.6px;">Personal & Financial Profile</div>
            <div style="font-size:12px;color:#64748B;margin-top:1px;">Enter applicant details to generate a credit assessment</div>
        </div>
    </div>
    """)

    with st.form("credit_form"):
        H("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid #F1F5F9;">Personal Information</div>""")

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1,
                                  help="Applicant's age in years")
        with c2:
            occupation = st.selectbox("Occupation", options=OCCUPATIONS, index=4)

        H("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1.5px;margin:20px 0 12px;padding-bottom:8px;border-bottom:1px solid #F1F5F9;">Financial Information</div>""")

        c3, c4 = st.columns(2)
        with c3:
            annual_income = st.number_input("Annual Income (₹)", min_value=0.0, value=600000.0, step=10000.0, format="%.0f")
        with c4:
            delayed_payments = st.number_input("Delayed Payments", min_value=0, max_value=50, value=1, step=1,
                                               help="Number of recorded late/missed payments")

        c5, c6 = st.columns(2)
        with c5:
            emi = st.number_input("Total Monthly EMI (₹)", min_value=0.0, value=5000.0, step=500.0, format="%.0f")
        with c6:
            outstanding_debt = st.number_input("Outstanding Debt (₹)", min_value=0.0, value=150000.0, step=5000.0, format="%.0f")

        monthly_balance = st.number_input(
            "Monthly Closing Balance (₹)", min_value=0.0, value=40000.0, step=1000.0, format="%.0f",
            help="Average monthly bank balance after all expenses")

        # ── live derived ratio panel ──
        mi        = annual_income / 12 if annual_income > 0 else 1
        emi_ratio = emi / mi           if mi > 0           else 0
        dti       = outstanding_debt / annual_income if annual_income > 0 else 0
        sav_pct   = min(100.0, monthly_balance / mi * 100) if mi > 0 else 0

        def _clr(bad): return "#EF4444" if bad else "#10B981"
        def _lbl(bad, good_txt, bad_txt): return bad_txt if bad else good_txt

        H(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:16px 18px;margin:18px 0;display:flex;gap:0;align-items:stretch;">
            <div style="flex:1;text-align:center;padding:0 10px;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">EMI / Income</div>
                <div style="font-size:22px;font-weight:900;color:{_clr(emi_ratio>0.4)};line-height:1;">{emi_ratio:.0%}</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:3px;">{_lbl(emi_ratio>0.4,"✓ Healthy","▲ High")}</div>
            </div>
            <div style="width:1px;background:#E2E8F0;"></div>
            <div style="flex:1;text-align:center;padding:0 10px;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Debt / Income</div>
                <div style="font-size:22px;font-weight:900;color:{_clr(dti>1.5)};line-height:1;">{dti:.2f}×</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:3px;">{_lbl(dti>1.5,"✓ Healthy","▲ High")}</div>
            </div>
            <div style="width:1px;background:#E2E8F0;"></div>
            <div style="flex:1;text-align:center;padding:0 10px;">
                <div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Savings Rate</div>
                <div style="font-size:22px;font-weight:900;color:{_clr(sav_pct<10)};line-height:1;">{sav_pct:.0f}%</div>
                <div style="font-size:10px;color:#94A3B8;margin-top:3px;">{_lbl(sav_pct<10,"✓ Healthy","▲ Low")}</div>
            </div>
        </div>
        """)

        H("""<div style="font-size:10px;color:#94A3B8;margin-bottom:14px;">
            7 model inputs: Age · Occupation · Annual Income · Delayed Payments · Monthly EMI · Outstanding Debt · Monthly Balance
        </div>""")

        b1, b2 = st.columns([2, 1])
        with b1:
            submitted = st.form_submit_button("▶  Run Credit Assessment", type="primary", use_container_width=True)
        with b2:
            reset = st.form_submit_button("Reset", type="secondary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# RIGHT — ASSESSMENT PANEL
# ─────────────────────────────────────────────────────────────────
with right_col:
    H("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;">
        <div style="width:30px;height:30px;background:#2563EB;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:800;color:#fff;flex-shrink:0;">2</div>
        <div>
            <div style="font-size:14px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.6px;">Live Credit Assessment</div>
            <div style="font-size:12px;color:#64748B;margin-top:1px;">Instant AI-powered risk analysis</div>
        </div>
    </div>
    """)

    # handle reset
    if reset and "result" in st.session_state:
        del st.session_state["result"]
        st.rerun()

    # handle submit
    if submitted:
        input_df              = build_input_df(age, occupation, annual_income, delayed_payments, emi, outstanding_debt, monthly_balance)
        prediction, probabilities = run_prediction(input_df)
        risk_score            = compute_risk_score(probabilities, class_names)
        recs                  = get_recommendations(prediction, annual_income, delayed_payments, emi, outstanding_debt, monthly_balance)
        prob_map              = {c: float(probabilities[i]) for i, c in enumerate(class_names)}

        st.session_state["result"] = {
            "prediction":    prediction,
            "probabilities": probabilities,
            "risk_score":    risk_score,
            "prob_map":      prob_map,
            "recs":          recs,
            "input_df":      input_df,
            "inputs": {
                "Age":               age,
                "Occupation":        occupation,
                "Annual Income":     f"₹{annual_income:,.0f}",
                "Delayed Payments":  int(delayed_payments),
                "Total Monthly EMI": f"₹{emi:,.0f}",
                "Outstanding Debt":  f"₹{outstanding_debt:,.0f}",
                "Monthly Balance":   f"₹{monthly_balance:,.0f}",
            },
        }

    # ── WAITING STATE ──
    if "result" not in st.session_state:
        H("""
        <div style="background:#FFFFFF;border:1.5px solid #E2E8F0;border-radius:14px;padding:64px 40px;text-align:center;">
            <div style="font-size:54px;margin-bottom:18px;">📊</div>
            <div style="font-size:18px;font-weight:800;color:#0F172A;margin-bottom:8px;">Awaiting Assessment</div>
            <div style="font-size:13px;color:#64748B;max-width:280px;margin:0 auto;line-height:1.75;">
                Complete the financial profile and click
                <span style="font-weight:800;color:#2563EB;">Run Credit Assessment</span>
                to generate an instant risk analysis.
            </div>
            <div style="display:flex;justify-content:center;gap:0;margin-top:32px;border:1px solid #E2E8F0;border-radius:10px;overflow:hidden;max-width:280px;margin-left:auto;margin-right:auto;">
                <div style="flex:1;padding:16px 12px;text-align:center;border-right:1px solid #E2E8F0;">
                    <div style="font-size:22px;font-weight:900;color:#0F172A;">3</div>
                    <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Classes</div>
                </div>
                <div style="flex:1;padding:16px 12px;text-align:center;border-right:1px solid #E2E8F0;">
                    <div style="font-size:22px;font-weight:900;color:#0F172A;">7</div>
                    <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Features</div>
                </div>
                <div style="flex:1;padding:16px 12px;text-align:center;">
                    <div style="font-size:22px;font-weight:900;color:#0F172A;">31K</div>
                    <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Records</div>
                </div>
            </div>
        </div>
        """)

    # ── RESULT STATE ──
    else:
        r    = st.session_state["result"]
        pred = r["prediction"]
        cfg  = RATING.get(pred, RATING["Standard"])
        risk = r["risk_score"]

        # Rating banner
        H(f"""
        <div style="background:linear-gradient(135deg,{cfg['bg']} 0%,#FFFFFF 100%);border:1.5px solid {cfg['border']};border-radius:14px;padding:24px 28px;margin-bottom:18px;">
            <div style="font-size:9px;font-weight:800;color:{cfg['dark']};text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Final Credit Assessment</div>
            <div style="font-size:54px;font-weight:900;color:{cfg['hex']};letter-spacing:-2.5px;line-height:1;margin-bottom:12px;">{pred.upper()}</div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                <div style="display:inline-block;background:rgba(255,255,255,0.85);border:1.5px solid {cfg['border']};border-radius:20px;padding:5px 16px;font-size:12px;font-weight:700;color:{cfg['dark']};">{pred} Credit Profile</div>
                <div style="display:inline-block;background:rgba(255,255,255,0.6);border:1px solid #E2E8F0;border-radius:20px;padding:5px 16px;font-size:12px;font-weight:600;color:#64748B;">Risk Score: <b style="color:{cfg['hex']};">{risk:.0f} / 100</b></div>
            </div>
        </div>
        """)

        # Gauge + probabilities
        gc, pc = st.columns(2)
        with gc:
            H("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;text-align:center;margin-bottom:4px;">Risk Gauge</div>""")
            if PLOTLY_OK:
                st.plotly_chart(make_gauge(risk, pred), use_container_width=True, config={"displayModeBar": False})
            H("""<div style="text-align:center;margin-top:-10px;font-size:10px;color:#94A3B8;">0 = Low Risk · 100 = Critical</div>""")

        with pc:
            H("""<div style="font-size:9px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;text-align:center;margin-bottom:4px;">Class Probabilities</div>""")
            if PLOTLY_OK:
                st.plotly_chart(make_prob_chart(r["probabilities"], class_names, pred),
                                use_container_width=True, config={"displayModeBar": False})

        # Confidence bar
        max_prob = float(max(r["probabilities"])) * 100
        H(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:14px 20px;display:flex;justify-content:space-between;align-items:center;">
            <div>
                <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;">Model Confidence</div>
                <div style="font-size:11px;color:#64748B;margin-top:2px;">Highest predicted class probability</div>
            </div>
            <div style="font-size:28px;font-weight:900;color:#0F172A;letter-spacing:-1px;">{max_prob:.1f}%</div>
        </div>
        """)

# ─────────────────────────────────────────────────────────────────
# ANALYTICS TABS  (visible only after an assessment)
# ─────────────────────────────────────────────────────────────────
if "result" in st.session_state:
    r    = st.session_state["result"]
    pred = r["prediction"]
    cfg  = RATING.get(pred, RATING["Standard"])

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "  Feature Impact  ",
        "  Recommendations  ",
        "  Model Performance  ",
        "  Input Summary  ",
        "  Export Report  ",
    ])

    # ── TAB 1: Feature Impact ──────────────────────────────────
    with tab1:
        ta, tb = st.columns([1.2, 0.85], gap="large")

        with ta:
            H("""
            <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Key Model Drivers</div>
            <div style="font-size:12px;color:#64748B;margin-bottom:18px;">Relative Gini-MDI importance learned by the Extra Trees ensemble across all calibrated estimators</div>
            """)
            if PLOTLY_OK:
                st.plotly_chart(make_importance_chart(importance_df),
                                use_container_width=True, config={"displayModeBar": False})
            H("""<div style="font-size:11px;color:#94A3B8;margin-top:4px;">
                ⚠️ Importance reflects model learning — it does not imply direct causation.
            </div>""")

        with tb:
            H("""<div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:16px;">Breakdown</div>""")
            for _, row in importance_df.iterrows():
                disp = FEAT_DISPLAY.get(row["Feature"], row["Feature"])
                pct  = row["Importance"] * 100
                bar  = int(pct / importance_df["Importance"].max() * 100)
                H(f"""
                <div style="margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:12px;font-weight:700;color:#374151;">{disp}</span>
                        <span style="font-size:12px;font-weight:800;color:#0F172A;">{pct:.1f}%</span>
                    </div>
                    <div style="background:#F1F5F9;border-radius:5px;height:7px;">
                        <div style="width:{bar}%;height:100%;background:#2563EB;border-radius:5px;"></div>
                    </div>
                </div>
                """)

    # ── TAB 2: Recommendations ────────────────────────────────
    with tab2:
        H("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Personalised Financial Guidance</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:22px;">Actionable recommendations based on this applicant's specific financial profile</div>
        """)

        bc = cfg["hex"]
        for title, body in r["recs"]:
            H(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid {bc};border-radius:10px;padding:18px 22px;margin-bottom:14px;">
                <div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:7px;">{title}</div>
                <div style="font-size:13px;color:#475569;line-height:1.75;">{body}</div>
            </div>
            """)

        H("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:14px 18px;margin-top:6px;">
            <div style="font-size:11px;color:#94A3B8;line-height:1.6;">
                ⚠️ Recommendations are generated algorithmically from submitted data for analytical purposes only. They do not constitute financial, legal, or credit advice.
            </div>
        </div>
        """)

    # ── TAB 3: Model Performance ──────────────────────────────
    with tab3:
        H("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Model Performance Metrics</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:22px;">Evaluated on a held-out test set — Extra Trees Classifier with isotonic calibration</div>
        """)

        m_cols = st.columns(3)
        for i, (label, val) in enumerate(MODEL_METRICS.items()):
            with m_cols[i % 3]:
                st.metric(label, f"{val:.2f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        H("""<div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:16px;">Per-Class Breakdown</div>""")

        c_cols = st.columns(3)
        for i, (cls, mets) in enumerate(CLASS_METRICS.items()):
            c = RATING.get(cls, {})
            with c_cols[i]:
                H(f"""
                <div style="background:#FFFFFF;border:1.5px solid {c.get('border','#E2E8F0')};border-radius:12px;padding:20px;text-align:center;">
                    <div style="font-size:11px;font-weight:800;color:{c.get('dark','#374151')};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:14px;">{cls}</div>
                    <div style="display:flex;justify-content:space-around;gap:4px;">
                        <div>
                            <div style="font-size:20px;font-weight:900;color:{c.get('hex','#374151')};">{mets['Precision']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Precision</div>
                        </div>
                        <div>
                            <div style="font-size:20px;font-weight:900;color:{c.get('hex','#374151')};">{mets['Recall']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Recall</div>
                        </div>
                        <div>
                            <div style="font-size:20px;font-weight:900;color:{c.get('hex','#374151')};">{mets['F1']:.1f}%</div>
                            <div style="font-size:9px;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">F1</div>
                        </div>
                    </div>
                </div>
                """)

        H("""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:16px 20px;margin-top:22px;">
            <div style="font-size:12px;color:#64748B;line-height:1.8;">
                <b style="color:#374151;">Methodology:</b> Final model selected via 10-fold cross-validated macro F1.
                Probability calibration applied post-training via isotonic regression on a held-out calibration split.
                Feature importance derived by averaging Gini-MDI scores across all calibrated base estimators.
            </div>
        </div>
        """)

    # ── TAB 4: Input Summary ──────────────────────────────────
    with tab4:
        H("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Assessment Input Summary</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:22px;">Exact values submitted to the model for this assessment session</div>
        """)

        s1, s2 = st.columns([1, 1], gap="large")

        with s1:
            H("""<div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Profile Details</div>""")
            rows_html = ""
            for k, v in r["inputs"].items():
                rows_html += f"""
                <div style="display:flex;justify-content:space-between;padding:10px 0;border-bottom:1px solid #F1F5F9;">
                    <span style="font-size:13px;color:#64748B;font-weight:500;">{k}</span>
                    <span style="font-size:13px;color:#0F172A;font-weight:700;">{v}</span>
                </div>
                """
            H(f"""<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:4px 18px;">{rows_html}</div>""")

        with s2:
            H("""<div style="font-size:11px;font-weight:800;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">Assessment Outcome</div>""")

            prob_map = r["prob_map"]
            color_m  = {"Good": "#10B981", "Standard": "#F59E0B", "Poor": "#EF4444"}

            H(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;padding:20px;">
                <div style="margin-bottom:18px;">
                    <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Credit Class</div>
                    <div style="font-size:30px;font-weight:900;color:{cfg['hex']};">{pred}</div>
                </div>
                <div style="margin-bottom:18px;">
                    <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Risk Score</div>
                    <div style="font-size:30px;font-weight:900;color:#0F172A;">{r['risk_score']:.0f}<span style="font-size:14px;color:#94A3B8;font-weight:600;"> / 100</span></div>
                </div>
                <div>
                    <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;">Class Probabilities</div>
            """)

            for cls in class_names:
                pct = prob_map.get(cls, 0) * 100
                clr = color_m.get(cls, "#6B7280")
                bold = "800" if cls == pred else "500"
                H(f"""
                <div style="margin-bottom:11px;">
                    <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                        <span style="font-size:12px;color:#374151;font-weight:{bold};">{cls}</span>
                        <span style="font-size:12px;font-weight:800;color:{clr};">{pct:.1f}%</span>
                    </div>
                    <div style="background:#F1F5F9;border-radius:4px;height:7px;">
                        <div style="width:{pct}%;height:100%;background:{clr};border-radius:4px;"></div>
                    </div>
                </div>
                """)

            H("</div></div>")

    # ── TAB 5: Export Report ──────────────────────────────────
    with tab5:
        H("""
        <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:4px;">Export Assessment Report</div>
        <div style="font-size:12px;color:#64748B;margin-bottom:24px;">Generate a detailed professional PDF report for this credit assessment</div>
        """)

        rc1, rc2 = st.columns([1, 1], gap="large")

        with rc1:
            H(f"""
            <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:26px;">
                <div style="font-size:13px;font-weight:800;color:#0F172A;margin-bottom:18px;">Report Contents</div>
                <div style="font-size:13px;color:#475569;line-height:2.4;">
                    ✓ &nbsp; Applicant financial profile summary<br>
                    ✓ &nbsp; Credit classification: <b style="color:{cfg['hex']};">{pred}</b><br>
                    ✓ &nbsp; Class probability breakdown<br>
                    ✓ &nbsp; Feature importance table<br>
                    ✓ &nbsp; Personalised recommendations<br>
                    ✓ &nbsp; Model performance metrics<br>
                    ✓ &nbsp; Methodology &amp; disclaimer
                </div>
            </div>
            """)

        with rc2:
            H("""
            <div style="background:#F8FAFC;border:1.5px dashed #CBD5E1;border-radius:12px;padding:32px;text-align:center;">
                <div style="font-size:38px;margin-bottom:14px;">📄</div>
                <div style="font-size:15px;font-weight:800;color:#0F172A;margin-bottom:6px;">PDF Credit Report</div>
                <div style="font-size:12px;color:#64748B;margin-bottom:22px;line-height:1.6;">Professional multi-page report with full assessment details</div>
            </div>
            """)

            if PDF_OK:
                try:
                    recs_clean = [(t.split(" ", 1)[-1] if t[0] in "🔔📉⚖️🏦✅📋📊" else t, b)
                                  for t, b in r["recs"]]
                    pdf_bytes = pdf_mod.generate_pdf(
                        input_df      = r["input_df"],
                        prediction    = r["prediction"],
                        probabilities = r["probabilities"],
                        class_names   = class_names,
                        recommendations = recs_clean,
                        importance_df = importance_df,
                    )
                    if pdf_bytes:
                        st.download_button(
                            label    = "⬇  Download PDF Report",
                            data     = io.BytesIO(pdf_bytes),
                            file_name= f"CrediX_{r['prediction']}_Assessment.pdf",
                            mime     = "application/pdf",
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(f"PDF generation failed: {exc}")
            else:
                H("""<div style="font-size:12px;color:#94A3B8;text-align:center;margin-top:10px;">PDF unavailable — ensure reportlab is installed.</div>""")
