import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CrediX | Credit Intelligence",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html):
    """
    Render custom HTML using Streamlit's native HTML renderer.
    This prevents HTML from appearing as Markdown/code.
    """
    st.html(html)


# ============================================================
# PREMIUM UI CSS
# ============================================================

render_html("""
<style>

:root {
    --bg: #070A12;
    --panel: #101521;
    --panel-2: #141B2E;
    --border: rgba(255,255,255,0.07);
    --muted: #7F899D;
    --text: #F4F7FB;
    --accent: #7188FF;
    --accent2: #835FFF;
}

.stApp {
    background:
        radial-gradient(
            circle at 5% 0%,
            rgba(87, 111, 255, 0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 95% 5%,
            rgba(131, 95, 255, 0.10),
            transparent 28%
        ),
        #070A12;
    color: var(--text);
}

.block-container {
    max-width: 1280px;
    padding-top: 32px;
    padding-bottom: 70px;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* ============================================================
   BRAND
============================================================ */

.brand-wrapper {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 25px;
}

.brand {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #F4F7FB;
}

.brand-accent {
    color: #7188FF;
}

.tagline {
    color: #747F94;
    font-size: 12px;
    margin-top: 3px;
}

.system-badge {
    padding: 7px 12px;
    border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.08);
    color: #747F94;
    font-size: 10px;
    letter-spacing: 0.7px;
    background: rgba(255,255,255,0.02);
}


/* ============================================================
   HERO
============================================================ */

.hero {
    padding: 42px;
    border-radius: 25px;

    background:
        radial-gradient(
            circle at 85% 15%,
            rgba(113,136,255,0.18),
            transparent 32%
        ),
        radial-gradient(
            circle at 60% 100%,
            rgba(131,95,255,0.07),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            rgba(19,27,47,0.98),
            rgba(8,12,22,0.98)
        );

    border: 1px solid rgba(113,136,255,0.12);

    box-shadow:
        0 30px 80px rgba(0,0,0,0.28);

    margin-bottom: 10px;
}

.hero-eyebrow {
    color: #7F8BA2;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 40px;
    line-height: 1.12;
    font-weight: 800;
    letter-spacing: -1.8px;
    color: #F7F9FC;
}

.hero-text {
    color: #8993A7;
    max-width: 690px;
    line-height: 1.75;
    font-size: 13px;
    margin-top: 17px;
}


/* ============================================================
   SECTIONS
============================================================ */

.section-title {
    font-size: 21px;
    font-weight: 750;
    color: #F1F4F9;
    margin-top: 34px;
    margin-bottom: 5px;
}

.section-description {
    color: #737E93;
    font-size: 12px;
    margin-bottom: 18px;
}


/* ============================================================
   RESULT CARD
============================================================ */

.result-card {
    min-height: 245px;
    padding: 31px;
    border-radius: 22px;

    background:
        radial-gradient(
            circle at 82% 20%,
            rgba(113,136,255,0.16),
            transparent 34%
        ),
        linear-gradient(
            135deg,
            rgba(21,29,50,0.98),
            rgba(10,14,25,0.98)
        );

    border: 1px solid rgba(113,136,255,0.13);

    box-shadow:
        0 18px 55px rgba(0,0,0,0.20);
}

.result-label {
    color: #7F8BA2;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}

.result-value {
    color: #F6F8FC;
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.8px;
    margin-top: 9px;
}

.result-description {
    color: #8C96A9;
    font-size: 12px;
    line-height: 1.7;
    margin-top: 13px;
}


/* ============================================================
   STATUS BADGES
============================================================ */

.status-good {
    display: inline-block;
    margin-top: 12px;
    padding: 7px 13px;

    border-radius: 100px;

    color: #9CE6BF;
    background: rgba(66,190,123,0.09);

    border: 1px solid rgba(66,190,123,0.18);

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.status-standard {
    display: inline-block;
    margin-top: 12px;
    padding: 7px 13px;

    border-radius: 100px;

    color: #D9C17B;
    background: rgba(213,176,70,0.09);

    border: 1px solid rgba(213,176,70,0.18);

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}

.status-poor {
    display: inline-block;
    margin-top: 12px;
    padding: 7px 13px;

    border-radius: 100px;

    color: #FF9D9D;
    background: rgba(220,70,70,0.09);

    border: 1px solid rgba(220,70,70,0.18);

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.5px;
}


/* ============================================================
   CONFIDENCE CARD
============================================================ */

.confidence-card {
    min-height: 245px;
    padding: 31px;

    border-radius: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(17,23,37,0.98),
            rgba(9,13,23,0.98)
        );

    border: 1px solid rgba(255,255,255,0.065);

    box-shadow:
        0 18px 55px rgba(0,0,0,0.18);
}

.card-label {
    color: #7F899D;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.1px;
}

.card-number {
    color: #F4F7FB;
    font-size: 34px;
    font-weight: 800;
    margin-top: 9px;
}

.card-description {
    color: #737D91;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 6px;
}

.model-name {
    color: #EDEFF5;
    font-size: 18px;
    font-weight: 700;
    margin-top: 6px;
}


/* ============================================================
   RECOMMENDATIONS
============================================================ */

.recommendation {
    padding: 20px;

    min-height: 105px;

    border-radius: 16px;

    background:
        linear-gradient(
            135deg,
            rgba(23,31,52,0.82),
            rgba(12,17,29,0.82)
        );

    border: 1px solid rgba(113,136,255,0.10);

    transition: all 0.2s ease;

    margin-bottom: 12px;
}

.recommendation:hover {
    border-color: rgba(113,136,255,0.25);
    transform: translateY(-1px);
}

.recommendation-title {
    color: #F2F5FA;
    font-size: 13px;
    font-weight: 700;
}

.recommendation-text {
    color: #8993A6;
    font-size: 11px;
    line-height: 1.7;
    margin-top: 7px;
}


/* ============================================================
   INPUTS
============================================================ */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {

    background: #101521 !important;

    border:
        1px solid rgba(255,255,255,0.08)
        !important;

    border-radius: 10px !important;
}

input {
    color: #F4F7FB !important;
}

label {
    color: #A7B0C0 !important;
    font-size: 12px !important;
}


/* ============================================================
   BUTTONS
============================================================ */

.stButton > button {

    width: 100%;

    min-height: 46px;

    border-radius: 11px;

    border: none;

    background:
        linear-gradient(
            135deg,
            #7188FF,
            #835FFF
        );

    color: white;

    font-weight: 700;

    box-shadow:
        0 10px 30px rgba(103,122,255,0.18);

    transition: all 0.2s ease;
}

.stButton > button:hover {

    border: none;

    transform: translateY(-1px);

    box-shadow:
        0 14px 35px rgba(103,122,255,0.28);
}


/* ============================================================
   METRICS
============================================================ */

[data-testid="stMetric"] {

    background:
        rgba(255,255,255,0.025);

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius: 14px;

    padding: 15px;
}


/* ============================================================
   EXPANDERS
============================================================ */

[data-testid="stExpander"] {

    background:
        rgba(255,255,255,0.018);

    border:
        1px solid rgba(255,255,255,0.06);

    border-radius: 14px;
}


/* ============================================================
   FOOTER
============================================================ */

.footer {
    margin-top: 45px;
    padding: 22px;

    text-align: center;

    color: #596275;

    font-size: 10px;

    border-top:
        1px solid rgba(255,255,255,0.06);
}

.footer-title {
    color: #747E91;
    font-size: 11px;
    font-weight: 600;
}

.footer-subtitle {
    margin-top: 8px;
    color: #505A6C;
}

</style>
""")


# ============================================================
# MODEL IMPORT
# ============================================================

try:

    from model import train_model, predict

except Exception as e:

    st.error(
        "The machine learning module could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    return train_model()


try:

    model, feature_names, label_encoder = load_ml_model()

except Exception as e:

    st.error(
        "The machine learning model could not be initialized."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

render_html("""
<div class="brand-wrapper">

    <div>

        <div class="brand">
            Credi<span class="brand-accent">X</span>
        </div>

        <div class="tagline">
            Explainable Machine Learning Credit Intelligence
        </div>

    </div>

    <div class="system-badge">
        AI CREDIT ANALYSIS
    </div>

</div>
""")


# ============================================================
# HERO
# ============================================================

render_html("""
<div class="hero">

    <div class="hero-eyebrow">
        PERSONAL CREDIT INTELLIGENCE
    </div>

    <div class="hero-title">
        Understand your credit.<br>
        Make better financial decisions.
    </div>

    <div class="hero-text">
        CrediX uses machine learning to analyse your financial
        profile and explain the factors influencing your predicted
        credit category. The platform combines prediction,
        explainability and personalised improvement guidance
        in one customer-focused experience.
    </div>

</div>
""")


# ============================================================
# FINANCIAL PROFILE
# ============================================================

render_html("""
<div class="section-title">
    Financial Profile
</div>

<div class="section-description">
    Enter your current financial information to generate your assessment.
</div>
""")


c1, c2, c3 = st.columns(3)


with c1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with c2:

    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )


with c3:

    num_loans = st.selectbox(
        "Number of Loans",
        list(range(0, 11)),
        index=2
    )


c4, c5, c6 = st.columns(3)


with c4:

    emi = st.number_input(
        "Monthly EMI",
        min_value=0.0,
        value=5000.0,
        step=500.0
    )


with c5:

    debt = st.number_input(
        "Outstanding Debt",
        min_value=0.0,
        value=15000.0,
        step=1000.0
    )


with c6:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=40000.0,
        step=1000.0
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# ANALYSE BUTTON
# ============================================================

analyse = st.button(
    "Analyse My Credit Profile"
)


# ============================================================
# ANALYSIS
# ============================================================

if analyse:

    input_df = pd.DataFrame([{

        "Age": int(age),

        "Income": float(income),

        "Num_Loans": int(num_loans),

        "EMI": float(emi),

        "Outstanding_Debt": float(debt),

        "Balance": float(balance)

    }])


    # ========================================================
    # PREDICTION
    # ========================================================

    try:

        prediction, probabilities = predict(
            model,
            input_df
        )

    except Exception as e:

        st.error(
            "Prediction could not be generated."
        )

        st.code(str(e))

        st.stop()


    # ========================================================
    # DECODE RESULT
    # ========================================================

    try:

        result = label_encoder.inverse_transform(
            prediction
        )[0]

    except Exception:

        result = str(
            prediction[0]
        )


    confidence = float(
        np.max(
            probabilities[0]
        ) * 100
    )


    # ========================================================
    # CREDIT ASSESSMENT
    # ========================================================

    render_html("""
    <div class="section-title">
        Credit Assessment
    </div>
    """)


    left, right = st.columns(
        [1.25, 1]
    )


    with left:

        result_lower = str(
            result
        ).lower()


        if result_lower == "good":

            status_class = "status-good"

            description = (
                "Your current financial profile is classified "
                "as relatively healthy."
            )

        elif result_lower == "standard":

            status_class = "status-standard"

            description = (
                "Your financial profile is moderate and has "
                "areas where improvement may be beneficial."
            )

        else:

            status_class = "status-poor"

            description = (
                "Your current profile indicates higher financial "
                "risk and may benefit from corrective actions."
            )


        render_html(
            f"""
            <div class="result-card">

                <div class="result-label">
                    PREDICTED CREDIT CATEGORY
                </div>

                <div class="result-value">
                    {result}
                </div>

                <div class="{status_class}">
                    {str(result).upper()}
                </div>

                <div class="result-description">
                    {description}
                </div>

            </div>
            """
        )


    with right:

        render_html(
            f"""
            <div class="confidence-card">

                <div class="card-label">
                    MODEL CONFIDENCE
                </div>

                <div class="card-number">
                    {confidence:.1f}%
                </div>

                <div class="card-description">
                    Confidence assigned by the machine learning
                    model to the predicted category.
                </div>

                <br>

                <div class="card-label">
                    MODEL
                </div>

                <div class="model-name">
                    Random Forest
                </div>

                <div class="card-description">
                    Ensemble machine learning classifier
                </div>

            </div>
            """
        )


    # ========================================================
    # NEXT BEST ACTIONS
    # ========================================================

    render_html("""
    <div class="section-title">
        Your Next Best Actions
    </div>

    <div class="section-description">
        Personalised suggestions based on the financial
        information used in your assessment.
    </div>
    """)


    recommendations = []


    # --------------------------------------------------------
    # DEBT
    # --------------------------------------------------------

    if income > 0 and debt > income * 0.35:

        recommendations.append(
            (
                "Reduce outstanding debt",

                "Your outstanding debt is relatively high "
                "compared with your annual income. Prioritising "
                "repayment of high-cost or outstanding debt can "
                "strengthen your overall financial profile."
            )
        )


    # --------------------------------------------------------
    # LOAN EXPOSURE
    # --------------------------------------------------------

    if num_loans >= 5:

        recommendations.append(
            (
                "Manage loan exposure",

                "You currently have several active loans. "
                "Avoiding unnecessary additional borrowing can "
                "reduce your overall credit exposure and improve "
                "financial stability."
            )
        )


    # --------------------------------------------------------
    # EMI
    # --------------------------------------------------------

    if income > 0:

        monthly_income = income / 12

        if emi > monthly_income * 0.35:

            recommendations.append(
                (
                    "Lower monthly EMI burden",

                    "Your monthly EMI represents a significant "
                    "portion of your estimated monthly income. "
                    "Reducing monthly repayment obligations can "
                    "improve your debt-to-income position."
                )
            )


    # --------------------------------------------------------
    # FINANCIAL BUFFER
    # --------------------------------------------------------

    if income > 0 and balance < income * 0.10:

        recommendations.append(
            (
                "Build a stronger financial buffer",

                "Your current balance is relatively low compared "
                "with your income. Building an emergency reserve "
                "can provide greater financial stability."
            )
        )


    # --------------------------------------------------------
    # POSITIVE PROFILE
    # --------------------------------------------------------

    if not recommendations:

        recommendations.append(
            (
                "Maintain your current profile",

                "Your current financial indicators do not show "
                "a major warning signal. Continue managing loans "
                "responsibly, maintain timely repayments and build "
                "healthy financial reserves."
            )
        )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    rec1, rec2 = st.columns(2)


    for i, recommendation in enumerate(
        recommendations
    ):

        title, text = recommendation

        target = (
            rec1
            if i % 2 == 0
            else rec2
        )


        with target:

            render_html(
                f"""
                <div class="recommendation">

                    <div class="recommendation-title">
                        {title}
                    </div>

                    <div class="recommendation-text">
                        {text}
                    </div>

                </div>
                """
            )


    # ========================================================
    # FINANCIAL SNAPSHOT
    # ========================================================

    render_html("""
    <div class="section-title">
        Financial Snapshot
    </div>

    <div class="section-description">
        A quick overview of the financial information used by the model.
    </div>
    """)


    m1, m2, m3, m4 = st.columns(4)


    with m1:

        st.metric(
            "Annual Income",
            f"{income:,.0f}"
        )


    with m2:

        st.metric(
            "Outstanding Debt",
            f"{debt:,.0f}"
        )


    with m3:

        st.metric(
            "Monthly EMI",
            f"{emi:,.0f}"
        )


    with m4:

        st.metric(
            "Active Loans",
            int(num_loans)
        )


    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    render_html("""
    <div class="section-title">
        Prediction Confidence
    </div>

    <div class="section-description">
        Probability assigned to each credit category by the model.
    </div>
    """)


    try:

        class_names = label_encoder.inverse_transform(
            model.classes_
        )


        probability_df = pd.DataFrame({

            "Credit Category":
                class_names,

            "Probability":
                probabilities[0] * 100

        })


        fig, ax = plt.subplots(
            figsize=(7, 3)
        )


        ax.bar(
            probability_df["Credit Category"],
            probability_df["Probability"]
        )


        ax.set_ylabel(
            "Probability (%)"
        )


        ax.set_ylim(
            0,
            100
        )


        ax.set_title(
            "Model Prediction Confidence"
        )


        plt.tight_layout()


        st.pyplot(
            fig,
            use_container_width=False
        )


        plt.close(fig)


    except Exception:

        st.info(
            "Prediction probability graph is unavailable."
        )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    render_html("""
    <div class="section-title">
        Explainable AI
    </div>

    <div class="section-description">
        Understand which financial factors influenced the prediction.
    </div>
    """)


    with st.expander(
        "View AI Explanation"
    ):

        try:

            import shap


            explainer = shap.TreeExplainer(
                model
            )


            shap_result = explainer(
                input_df
            )


            shap_values = shap_result.values


            if len(
                shap_values.shape
            ) == 3:

                class_index = int(
                    prediction[0]
                )

                impacts = shap_values[
                    0,
                    :,
                    class_index
                ]

            else:

                impacts = shap_values[0]


            explanation_df = pd.DataFrame({

                "Feature":
                    feature_names,

                "Impact":
                    impacts

            })


            explanation_df[
                "Absolute Impact"
            ] = (
                explanation_df["Impact"].abs()
            )


            explanation_df = (
                explanation_df
                .sort_values(
                    "Absolute Impact",
                    ascending=False
                )
                .head(6)
            )


            explanation_df[
                "Direction"
            ] = np.where(

                explanation_df["Impact"] >= 0,

                "Supports prediction",

                "Opposes prediction"
            )


            st.dataframe(
                explanation_df[
                    [
                        "Feature",
                        "Impact",
                        "Direction"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )


        except Exception as e:

            st.info(
                "Detailed SHAP explanation is temporarily unavailable."
            )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    with st.expander(
        "Model Performance"
    ):

        st.write(
            "Algorithm: Random Forest Classifier"
        )

        st.write(
            f"Input features: {len(feature_names)}"
        )

        st.write(
            "Prediction type: Multi-class classification"
        )

        st.write(
            "Explainability method: SHAP"
        )

        st.write(
            "Evaluation metrics: Accuracy, Precision, Recall and F1-Score"
        )


    # ========================================================
    # PDF REPORT
    # ========================================================

    render_html("""
    <div class="section-title">
        Detailed Credit Report
    </div>

    <div class="section-description">
        Generate a downloadable report containing your assessment,
        financial profile and personalised recommendations.
    </div>
    """)


    if st.button(
        "Generate Detailed PDF Report"
    ):

        try:

            from pdf_report import generate_pdf


            pdf_result = generate_pdf(
                input_df,
                result,
                probabilities[0],
                recommendations
            )


            if isinstance(
                pdf_result,
                tuple
            ):

                pdf_bytes = pdf_result[0]

            else:

                pdf_bytes = pdf_result


            st.download_button(
                label="Download Credit Report",
                data=pdf_bytes,
                file_name="CrediX_Credit_Report.pdf",
                mime="application/pdf"
            )


        except Exception as e:

            st.warning(
                "PDF report could not be generated."
            )

            st.code(
                str(e)
            )


# ============================================================
# FOOTER
# ============================================================

render_html("""
<div class="footer">

    <div class="footer-title">
        CrediX — Explainable Machine Learning Credit Intelligence
    </div>

    <div class="footer-subtitle">
        Analytical and educational decision-support system.
        Predictions should not be treated as financial advice.
    </div>

</div>
""")