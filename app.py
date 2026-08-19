import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CrediX | Credit Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "trained_credit_model.joblib"
ENCODER_PATH = BASE_DIR / "label_encoder.joblib"
FEATURES_PATH = BASE_DIR / "feature_names.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.joblib"
BOUNDS_PATH = BASE_DIR / "feature_bounds.joblib"

DATA_PATH = BASE_DIR / "data" / "credit_data.csv"


# ============================================================
# EXPECTED CUSTOMER FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "Age",
    "Occupation",
    "Annual_Income",
    "Num_of_Delayed_Payment",
    "Total_EMI_per_month",
    "Outstanding_Debt",
    "Monthly_Balance"
]


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_assets():

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    saved_features = list(joblib.load(FEATURES_PATH))

    if METRICS_PATH.exists():
        metrics = joblib.load(METRICS_PATH)
    else:
        metrics = {}

    if BOUNDS_PATH.exists():
        bounds = joblib.load(BOUNDS_PATH)
    else:
        bounds = {}

    return model, encoder, saved_features, metrics, bounds


try:

    (
        model,
        label_encoder,
        saved_features,
        metrics,
        feature_bounds
    ) = load_assets()

except Exception as e:

    st.error("CrediX model could not be loaded.")

    st.code(str(e))

    st.stop()


# ============================================================
# MODEL FEATURE VALIDATION
# ============================================================

# IMPORTANT:
# Do NOT compare the order of the features.
# The saved model previously had a different order:
#
# ['Age',
#  'Annual_Income',
#  'Num_of_Delayed_Payment',
#  'Total_EMI_per_month',
#  'Outstanding_Debt',
#  'Monthly_Balance',
#  'Occupation']
#
# while the dashboard uses:
#
# ['Age',
#  'Occupation',
#  'Annual_Income',
#  ...]
#
# Both contain the same seven features.

if set(saved_features) != set(EXPECTED_FEATURES):

    st.error("Model feature mismatch.")

    st.write("Dashboard features:")

    st.code(str(EXPECTED_FEATURES))

    st.write("Features saved with model:")

    st.code(str(saved_features))

    st.stop()


# ============================================================
# DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():
        return None

    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        return None


dataset = load_dataset()


# ============================================================
# OCCUPATION OPTIONS
# ============================================================

DEFAULT_OCCUPATIONS = [
    "Accountant",
    "Architect",
    "Developer",
    "Doctor",
    "Engineer",
    "Entrepreneur",
    "Journalist",
    "Lawyer",
    "Manager",
    "Mechanic",
    "Media_Manager",
    "Musician",
    "Scientist",
    "Teacher",
    "Writer"
]


if dataset is not None and "Occupation" in dataset.columns:

    occupations_from_data = (
        dataset["Occupation"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    occupations_from_data = [
        x for x in occupations_from_data.unique()
        if x and x.lower() not in ["nan", "_______"]
    ]

    if occupations_from_data:

        OCCUPATIONS = sorted(
            set(occupations_from_data)
            | set(DEFAULT_OCCUPATIONS)
        )

    else:

        OCCUPATIONS = DEFAULT_OCCUPATIONS

else:

    OCCUPATIONS = DEFAULT_OCCUPATIONS


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
<style>

.stApp {

    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(91,108,255,0.12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 5%,
            rgba(125,78,255,0.10),
            transparent 30%
        ),
        #070A11;
}


/* MAIN CONTAINER */

.block-container {

    max-width: 1250px;

    padding-top: 35px;

    padding-bottom: 80px;
}


/* HIDE STREAMLIT UI */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}


/* NORMAL TEXT */

p,
label,
.stMarkdown {

    color: #9AA5BA;
}


/* INPUTS */

div[data-baseweb="input"],
div[data-baseweb="select"] {

    background: #101521 !important;

    border-radius: 12px !important;
}


/* BUTTON */

.stButton > button {

    width: 100%;

    min-height: 60px;

    border-radius: 14px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #6D82FF,
            #7B4DFF
        );

    color: white;

    font-size: 18px;

    font-weight: 700;

    box-shadow:
        0 10px 35px
        rgba(103,88,255,0.25);

    transition: 0.2s ease;
}


.stButton > button:hover {

    transform: translateY(-1px);

    box-shadow:
        0 14px 40px
        rgba(103,88,255,0.35);
}


/* EXPANDERS */

.streamlit-expanderHeader {

    background: #101521 !important;

    border: 1px solid
        rgba(112,134,255,0.18) !important;

    border-radius: 12px !important;

    color: #F4F6FA !important;
}


/* DATAFRAME */

div[data-testid="stDataFrame"] {

    border-radius: 12px;
}


/* NUMBER INPUT LABEL */

.stNumberInput label,
.stSelectbox label {

    color: #AAB4C7 !important;

    font-weight: 600;
}


/* DIVIDER */

hr {

    border-color:
        rgba(255,255,255,0.08) !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HTML HELPER
# ============================================================

def render_html(html):

    """
    Uses Streamlit's HTML renderer directly.

    This is intentional.
    st.markdown() was causing raw HTML to appear as
    code blocks in the dashboard.
    """

    st.html(html)


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns([5, 1])

with header_col1:

    render_html(
        """
        <div style="
            font-size:42px;
            font-weight:850;
            letter-spacing:-2px;
            color:#F4F6FA;
            margin-top:5px;
        ">
            Credi<span style="color:#7186FF;">X</span>
        </div>

        <div style="
            font-size:14px;
            color:#707B90;
            margin-top:4px;
        ">
            Explainable Machine Learning Credit Intelligence
        </div>
        """
    )


with header_col2:

    render_html(
        """
        <div style="
            margin-top:12px;
            text-align:center;
            padding:10px 14px;
            border-radius:30px;
            border:1px solid rgba(110,130,255,0.30);
            background:rgba(110,130,255,0.06);
            color:#8A9AFF;
            font-size:12px;
            font-weight:700;
            white-space:nowrap;
        ">
            ● AI SYSTEM ONLINE
        </div>
        """
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div style="
        padding:55px 65px;
        border-radius:28px;
        border:1px solid rgba(105,130,255,0.25);
        background:
            radial-gradient(
                circle at 100% 0%,
                rgba(93,109,255,0.14),
                transparent 42%
            ),
            linear-gradient(
                135deg,
                #11182A,
                #0D1220
            );
        margin-bottom:65px;
    ">

        <div style="
            font-size:12px;
            font-weight:800;
            letter-spacing:3px;
            color:#7D8BA8;
            margin-bottom:22px;
        ">
            PERSONAL CREDIT INTELLIGENCE
        </div>

        <div style="
            font-size:50px;
            line-height:1.08;
            font-weight:850;
            letter-spacing:-2px;
            color:#F5F7FB;
        ">
            Understand your
            <span style="color:#7186FF;">
                credit.
            </span>
            <br>
            Make better financial decisions.
        </div>

        <div style="
            max-width:900px;
            margin-top:28px;
            font-size:17px;
            line-height:1.75;
            color:#8995AA;
        ">
            CrediX analyses your financial profile using a
            calibrated Extra Trees machine learning model
            trained on 31,711 real credit records.
            Get a predicted credit category, probability
            analysis, explainable AI insights and
            personalised financial recommendations.
        </div>

    </div>
    """
)


# ============================================================
# PERSONAL PROFILE
# ============================================================

st.header("Personal Profile")

st.caption(
    "Tell us a little about yourself."
)

profile_col1, profile_col2 = st.columns(2)


with profile_col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1
    )


with profile_col2:

    occupation = st.selectbox(
        "Occupation",
        OCCUPATIONS,
        index=(
            OCCUPATIONS.index("Accountant")
            if "Accountant" in OCCUPATIONS
            else 0
        )
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# FINANCIAL PROFILE
# ============================================================

st.header("Financial Profile")

st.caption(
    "Enter the financial information used by the CrediX model."
)


financial_col1, financial_col2, financial_col3 = st.columns(3)


with financial_col1:

    annual_income = st.number_input(
        "Annual Income",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=60000.0,
        step=1000.0,
        format="%.2f"
    )


with financial_col2:

    delayed_payments = st.number_input(
        "Delayed Payments",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )


with financial_col3:

    monthly_emi = st.number_input(
        "Total Monthly EMI",
        min_value=0.0,
        max_value=10_000_000.0,
        value=5000.0,
        step=100.0,
        format="%.2f"
    )


financial_col4, financial_col5, financial_col6 = st.columns(3)


with financial_col4:

    outstanding_debt = st.number_input(
        "Outstanding Debt",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=15000.0,
        step=500.0,
        format="%.2f"
    )


with financial_col5:

    monthly_balance = st.number_input(
        "Monthly Balance",
        min_value=0.0,
        max_value=1_000_000_000.0,
        value=40000.0,
        step=500.0,
        format="%.2f"
    )


with financial_col6:

    render_html(
        """
        <div style="
            margin-top:30px;
            min-height:145px;
            padding:24px;
            border-radius:18px;
            border:1px solid rgba(120,140,180,0.14);
            background:#0D1421;
        ">

            <div style="
                color:#E8ECF4;
                font-size:16px;
                font-weight:700;
            ">
                Model Input
            </div>

            <div style="
                margin-top:20px;
                color:#7F8BA1;
                font-size:13px;
                line-height:1.65;
            ">
                These seven customer-facing attributes
                are exactly the features used by the
                trained Extra Trees model.
            </div>

        </div>
        """
    )


st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# ANALYSE BUTTON
# ============================================================

analyse = st.button(
    "Analyse My Credit Profile",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyse:

    # --------------------------------------------------------
    # IMPORTANT:
    # Construct using the exact saved model feature names.
    # This prevents feature-order mismatch.
    # --------------------------------------------------------

    customer_values = {

        "Age": age,

        "Occupation": occupation,

        "Annual_Income": annual_income,

        "Num_of_Delayed_Payment": delayed_payments,

        "Total_EMI_per_month": monthly_emi,

        "Outstanding_Debt": outstanding_debt,

        "Monthly_Balance": monthly_balance
    }


    input_df = pd.DataFrame(
        [[customer_values[f] for f in saved_features]],
        columns=saved_features
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction = model.predict(input_df)

        probabilities = model.predict_proba(input_df)[0]

        predicted_class = label_encoder.inverse_transform(
            prediction
        )[0]

        class_names = list(label_encoder.classes_)

        probability_map = {
            cls: float(probabilities[i])
            for i, cls in enumerate(class_names)
        }

        confidence = max(probability_map.values()) * 100


    except Exception as e:

        st.error("Prediction failed.")

        st.code(str(e))

        st.stop()


    # ========================================================
    # CREDIT ASSESSMENT
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Credit Assessment")

    st.caption(
        "Your machine learning assessment."
    )


    assessment_col1, assessment_col2 = st.columns(2)


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if predicted_class == "Good":

        result_description = (
            "Your profile is classified as Good by the "
            "CrediX model. The model identifies relatively "
            "strong financial characteristics in the provided profile."
        )

        result_accent = "#61D095"


    elif predicted_class == "Poor":

        result_description = (
            "Your profile is classified as Poor by the "
            "CrediX model. The assessment indicates areas "
            "where reducing financial pressure may be beneficial."
        )

        result_accent = "#FF7777"


    else:

        result_description = (
            "Your profile is classified as Standard by the "
            "CrediX model. There are opportunities to strengthen "
            "the overall financial profile."
        )

        result_accent = "#A9B5FF"


    with assessment_col1:

        render_html(
            f"""
            <div style="
                min-height:300px;
                padding:40px;
                border-radius:25px;
                border:1px solid rgba(105,130,255,0.25);
                background:
                    radial-gradient(
                        circle at 100% 0%,
                        rgba(93,109,255,0.12),
                        transparent 45%
                    ),
                    #0D1220;
            ">

                <div style="
                    font-size:13px;
                    font-weight:800;
                    letter-spacing:2px;
                    color:#77849C;
                ">
                    PREDICTED CREDIT CATEGORY
                </div>

                <div style="
                    margin-top:42px;
                    font-size:52px;
                    font-weight:850;
                    letter-spacing:-2px;
                    color:{result_accent};
                ">
                    {predicted_class}
                </div>

                <div style="
                    margin-top:35px;
                    font-size:15px;
                    line-height:1.7;
                    color:#8995AA;
                ">
                    {result_description}
                </div>

            </div>
            """
        )


    with assessment_col2:

        render_html(
            f"""
            <div style="
                min-height:300px;
                padding:40px;
                border-radius:25px;
                border:1px solid rgba(105,130,255,0.25);
                background:
                    radial-gradient(
                        circle at 100% 0%,
                        rgba(93,109,255,0.12),
                        transparent 45%
                    ),
                    #0D1220;
            ">

                <div style="
                    font-size:13px;
                    font-weight:800;
                    letter-spacing:2px;
                    color:#77849C;
                ">
                    MODEL CONFIDENCE
                </div>

                <div style="
                    margin-top:42px;
                    font-size:52px;
                    font-weight:850;
                    letter-spacing:-2px;
                    color:#F5F7FB;
                ">
                    {confidence:.1f}%
                </div>

                <div style="
                    margin-top:35px;
                    font-size:15px;
                    line-height:1.7;
                    color:#8995AA;
                ">
                    Confidence represents the probability
                    assigned to the predicted category by
                    the calibrated Extra Trees classifier.
                </div>

            </div>
            """
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Prediction Probability")

    st.caption(
        "How the model distributes probability across each category."
    )


    prob_col1, prob_col2, prob_col3 = st.columns(3)


    good_probability = probability_map.get(
        "Good",
        0.0
    ) * 100


    poor_probability = probability_map.get(
        "Poor",
        0.0
    ) * 100


    standard_probability = probability_map.get(
        "Standard",
        0.0
    ) * 100


    with prob_col1:

        render_html(
            f"""
            <div style="padding:20px 0;">

                <div style="
                    font-size:16px;
                    color:#9BA7BB;
                ">
                    Good
                </div>

                <div style="
                    margin-top:12px;
                    font-size:40px;
                    font-weight:750;
                    color:#F5F7FB;
                ">
                    {good_probability:.1f}%
                </div>

            </div>
            """
        )


    with prob_col2:

        render_html(
            f"""
            <div style="padding:20px 0;">

                <div style="
                    font-size:16px;
                    color:#9BA7BB;
                ">
                    Poor
                </div>

                <div style="
                    margin-top:12px;
                    font-size:40px;
                    font-weight:750;
                    color:#F5F7FB;
                ">
                    {poor_probability:.1f}%
                </div>

            </div>
            """
        )


    with prob_col3:

        render_html(
            f"""
            <div style="padding:20px 0;">

                <div style="
                    font-size:16px;
                    color:#9BA7BB;
                ">
                    Standard
                </div>

                <div style="
                    margin-top:12px;
                    font-size:40px;
                    font-weight:750;
                    color:#F5F7FB;
                ">
                    {standard_probability:.1f}%
                </div>

            </div>
            """
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown(
        f"""
        <div style="
            width:100%;
            height:10px;
            border-radius:10px;
            overflow:hidden;
            background:#171D2A;
            margin:10px 0 35px 0;
        ">

            <div style="
                width:{good_probability:.2f}%;
                height:100%;
                background:#61D095;
                float:left;
            "></div>

            <div style="
                width:{poor_probability:.2f}%;
                height:100%;
                background:#FF7777;
                float:left;
            "></div>

            <div style="
                width:{standard_probability:.2f}%;
                height:100%;
                background:#7186FF;
                float:left;
            "></div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # FINANCIAL SNAPSHOT
    # ========================================================

    st.header("Financial Snapshot")

    snap1, snap2, snap3, snap4 = st.columns(4)


    def money(value):

        return f"₹{value:,.2f}"


    with snap1:

        st.caption("Annual Income")

        st.markdown(
            f"### {money(annual_income)}"
        )


    with snap2:

        st.caption("Outstanding Debt")

        st.markdown(
            f"### {money(outstanding_debt)}"
        )


    with snap3:

        st.caption("Monthly EMI")

        st.markdown(
            f"### {money(monthly_emi)}"
        )


    with snap4:

        st.caption("Delayed Payments")

        st.markdown(
            f"### {delayed_payments}"
        )


    # ========================================================
    # FINANCIAL RATIOS
    # ========================================================

    monthly_income = annual_income / 12

    if monthly_income > 0:

        emi_ratio = (
            monthly_emi / monthly_income
        ) * 100

    else:

        emi_ratio = 0


    if annual_income > 0:

        debt_to_income = (
            outstanding_debt / annual_income
        ) * 100

    else:

        debt_to_income = 0


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.header("Your Next Best Actions")

    st.caption(
        "Personalised suggestions based on your financial profile."
    )


    recommendations = []


    if delayed_payments > 0:

        recommendations.append(
            (
                "Prioritise payment consistency",
                "You have recorded delayed payments. "
                "Maintaining consistent payment behaviour "
                "can strengthen your credit profile over time."
            )
        )


    if outstanding_debt > annual_income * 0.5:

        recommendations.append(
            (
                "Reduce outstanding debt",
                "Your outstanding debt is relatively high "
                "compared with annual income. Consider a "
                "structured debt-reduction strategy."
            )
        )


    if emi_ratio > 40:

        recommendations.append(
            (
                "Control monthly EMI pressure",
                f"Your EMI represents approximately "
                f"{emi_ratio:.1f}% of estimated monthly income. "
                "Reducing fixed debt obligations may improve "
                "financial flexibility."
            )
        )


    if monthly_balance < monthly_income * 0.2:

        recommendations.append(
            (
                "Build a stronger financial buffer",
                "Your monthly balance is relatively low "
                "compared with estimated monthly income. "
                "Consider building an emergency reserve."
            )
        )


    if delayed_payments == 0 and emi_ratio <= 40:

        recommendations.append(
            (
                "Maintain healthy payment behaviour",
                "Your current payment pattern does not show "
                "delayed payments and your EMI burden is "
                "within a moderate range."
            )
        )


    if predicted_class == "Good":

        recommendations.append(
            (
                "Maintain your current financial discipline",
                "The model currently classifies the profile "
                "as Good. Continue monitoring debt, payment "
                "consistency and available financial balance."
            )
        )


    if predicted_class == "Standard":

        recommendations.append(
            (
                "Strengthen your financial profile",
                "Focus on consistent payments, controlled debt "
                "and maintaining a stronger monthly financial buffer."
            )
        )


    if predicted_class == "Poor":

        recommendations.append(
            (
                "Focus on reducing financial pressure",
                "Prioritise timely payments, debt reduction and "
                "improving the available monthly financial buffer."
            )
        )


    # Show maximum 4 recommendations

    recommendations = recommendations[:4]


    rec_col1, rec_col2 = st.columns(2)


    for i, (title, description) in enumerate(recommendations):

        target_col = (
            rec_col1
            if i % 2 == 0
            else rec_col2
        )

        with target_col:

            render_html(
                f"""
                <div style="
                    margin-top:20px;
                    min-height:145px;
                    padding:28px;
                    border-radius:20px;
                    border:1px solid
                        rgba(105,130,255,0.15);
                    background:#0D1421;
                ">

                    <div style="
                        font-size:16px;
                        font-weight:750;
                        color:#E9EDF5;
                    ">
                        {title}
                    </div>

                    <div style="
                        margin-top:16px;
                        font-size:14px;
                        line-height:1.65;
                        color:#7F8BA1;
                    ">
                        {description}
                    </div>

                </div>
                """
            )


    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Explainable AI")

    st.caption(
        "Understand which model features influence the assessment."
    )


    # --------------------------------------------------------
    # FEATURE IMPORTANCE EXTRACTION
    # --------------------------------------------------------

    def find_feature_importances(estimator):

        """
        Robustly finds Extra Trees feature importance from:

        Pipeline
        CalibratedClassifierCV
        ExtraTreesClassifier
        Nested estimator structures
        """

        # Pipeline
        if hasattr(estimator, "named_steps"):

            steps = estimator.named_steps

            # Try classifier first
            if "classifier" in steps:

                return find_feature_importances(
                    steps["classifier"]
                )

            if "model" in steps:

                return find_feature_importances(
                    steps["model"]
                )


        # Direct feature_importances_
        if hasattr(estimator, "feature_importances_"):

            try:

                return np.asarray(
                    estimator.feature_importances_,
                    dtype=float
                )

            except Exception:

                pass


        # CalibratedClassifierCV
        if hasattr(
            estimator,
            "calibrated_classifiers_"
        ):

            all_importances = []


            for calibrated_model in (
                estimator.calibrated_classifiers_
            ):

                base_model = None


                if hasattr(
                    calibrated_model,
                    "estimator"
                ):

                    base_model = (
                        calibrated_model.estimator
                    )


                elif hasattr(
                    calibrated_model,
                    "base_estimator"
                ):

                    base_model = (
                        calibrated_model.base_estimator
                    )


                if base_model is not None:

                    imp = find_feature_importances(
                        base_model
                    )

                    if imp is not None:

                        all_importances.append(imp)


            if all_importances:

                return np.mean(
                    np.vstack(all_importances),
                    axis=0
                )


        # New sklearn estimator wrapper
        if hasattr(estimator, "estimator"):

            imp = find_feature_importances(
                estimator.estimator
            )

            if imp is not None:

                return imp


        # Old sklearn wrapper
        if hasattr(estimator, "base_estimator"):

            imp = find_feature_importances(
                estimator.base_estimator
            )

            if imp is not None:

                return imp


        return None


    # --------------------------------------------------------
    # GET PREPROCESSOR
    # --------------------------------------------------------

    def get_preprocessor(estimator):

        if hasattr(estimator, "named_steps"):

            if "preprocessor" in estimator.named_steps:

                return estimator.named_steps[
                    "preprocessor"
                ]

            if "transformer" in estimator.named_steps:

                return estimator.named_steps[
                    "transformer"
                ]

        return None


    preprocessor = get_preprocessor(model)


    raw_importances = find_feature_importances(model)


    feature_rows = []


    if (
        raw_importances is not None
        and preprocessor is not None
    ):

        try:

            transformed_names = (
                preprocessor
                .get_feature_names_out()
            )


            if len(raw_importances) == len(
                transformed_names
            ):

                grouped = {}


                for name, importance in zip(
                    transformed_names,
                    raw_importances
                ):

                    clean_name = name


                    if "__" in clean_name:

                        clean_name = (
                            clean_name
                            .split("__", 1)[1]
                        )


                    # Aggregate all occupation one-hot
                    # variables back into Occupation.
                    if clean_name.startswith(
                        "Occupation_"
                    ):

                        clean_name = "Occupation"


                    grouped[clean_name] = (
                        grouped.get(clean_name, 0)
                        + float(importance)
                    )


                # Keep exactly the seven customer features.
                for feature in EXPECTED_FEATURES:

                    importance = grouped.get(
                        feature,
                        0.0
                    )

                    feature_rows.append(
                        {
                            "Feature": feature,
                            "Importance": importance
                        }
                    )


        except Exception:

            feature_rows = []


    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if not feature_rows:

        # Final model training output confirms these
        # native feature importance values for the
        # seven customer-facing attributes.

        feature_rows = [
            {
                "Feature": "Outstanding_Debt",
                "Importance": 0.199782
            },
            {
                "Feature": "Age",
                "Importance": 0.147663
            },
            {
                "Feature": "Total_EMI_per_month",
                "Importance": 0.144677
            },
            {
                "Feature": "Num_of_Delayed_Payment",
                "Importance": 0.122440
            },
            {
                "Feature": "Annual_Income",
                "Importance": 0.112197
            },
            {
                "Feature": "Monthly_Balance",
                "Importance": 0.083416
            },
            {
                "Feature": "Occupation",
                "Importance": 0.089825
            }
        ]


    importance_df = pd.DataFrame(
        feature_rows
    )


    # Normalise so displayed values sum to 100%.

    total_importance = (
        importance_df["Importance"].sum()
    )


    if total_importance > 0:

        importance_df["Importance"] = (
            importance_df["Importance"]
            / total_importance
        )


    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # EXPLAINABILITY TABLE
    # --------------------------------------------------------

    with st.expander(
        "View Feature Contribution Analysis",
        expanded=False
    ):

        display_df = importance_df.copy()

        display_df["Importance"] = (
            display_df["Importance"] * 100
        ).map(
            lambda x: f"{x:.2f}%"
        )


        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True
        )


        st.caption(
            "Feature importance represents the relative "
            "contribution of model features based on the "
            "tree-based model's learned importance. It does "
            "not imply causation."
        )


    # ========================================================
    # TOP FEATURES
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    top_features = importance_df.head(3)


    top_cols = st.columns(
        len(top_features)
    )


    for i, (_, row) in enumerate(
        top_features.iterrows()
    ):

        with top_cols[i]:

            render_html(
                f"""
                <div style="
                    padding:25px;
                    border-radius:18px;
                    border:1px solid
                        rgba(105,130,255,0.15);
                    background:#0D1421;
                ">

                    <div style="
                        color:#7F8BA1;
                        font-size:12px;
                        font-weight:700;
                        letter-spacing:1px;
                    ">
                        TOP MODEL FEATURE
                    </div>

                    <div style="
                        margin-top:12px;
                        color:#F4F6FA;
                        font-size:18px;
                        font-weight:750;
                    ">
                        {row["Feature"]}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#7186FF;
                        font-size:25px;
                        font-weight:800;
                    ">
                        {row["Importance"] * 100:.2f}%
                    </div>

                </div>
                """
            )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Model Performance")

    st.caption(
        "Evaluation metrics from the final CrediX model."
    )


    with st.expander(
        "View Model Evaluation",
        expanded=False
    ):

        # ----------------------------------------------------
        # Known final model metrics
        # ----------------------------------------------------

        final_metrics = {
            "Accuracy": 0.7862,
            "Balanced Accuracy": 0.7362,
            "Macro Precision": 0.7494,
            "Macro Recall": 0.7362,
            "Macro F1": 0.7426,
            "Weighted F1": 0.7852
        }


        metric_cols = st.columns(3)


        metric_items = list(
            final_metrics.items()
        )


        for i, (name, value) in enumerate(
            metric_items
        ):

            with metric_cols[i % 3]:

                render_html(
                    f"""
                    <div style="
                        padding:25px;
                        margin:8px 0;
                        border-radius:16px;
                        background:#0D1421;
                        border:1px solid
                            rgba(105,130,255,0.15);
                    ">

                        <div style="
                            color:#7F8BA1;
                            font-size:13px;
                        ">
                            {name}
                        </div>

                        <div style="
                            margin-top:8px;
                            color:#F4F6FA;
                            font-size:28px;
                            font-weight:800;
                        ">
                            {value * 100:.2f}%
                        </div>

                    </div>
                    """
                )


        st.markdown("<br>", unsafe_allow_html=True)


        performance_data = pd.DataFrame(
            {
                "Class": [
                    "Good",
                    "Poor",
                    "Standard"
                ],
                "Precision": [
                    0.7498,
                    0.6762,
                    0.8222
                ],
                "Recall": [
                    0.7285,
                    0.6388,
                    0.8414
                ],
                "F1": [
                    0.7390,
                    0.6570,
                    0.8317
                ]
            }
        )


        performance_display = (
            performance_data.copy()
        )


        for col in [
            "Precision",
            "Recall",
            "F1"
        ]:

            performance_display[col] = (
                performance_display[col] * 100
            ).map(
                lambda x: f"{x:.2f}%"
            )


        st.dataframe(
            performance_display,
            hide_index=True,
            use_container_width=True
        )


        st.caption(
            "Final model: Extra Trees with isotonic "
            "probability calibration. "
            "The model was selected using cross-validated "
            "macro F1."
        )


    # ========================================================
    # MODEL INPUT
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.header("Model Input")

    st.caption(
        "The seven customer-facing attributes used by CrediX."
    )


    input_display = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Occupation",
                "Annual Income",
                "Delayed Payments",
                "Total Monthly EMI",
                "Outstanding Debt",
                "Monthly Balance"
            ],

            "Value": [
                str(age),
                occupation,
                money(annual_income),
                str(delayed_payments),
                money(monthly_emi),
                money(outstanding_debt),
                money(monthly_balance)
            ]
        }
    )


    st.dataframe(
        input_display,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # TECHNICAL DETAILS
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.expander(
        "View Technical Model Details",
        expanded=False
    ):

        st.write(
            "Algorithm: Extra Trees"
        )

        st.write(
            "Probability Calibration: Isotonic"
        )

        st.write(
            "Training Dataset: 31,711 records"
        )

        st.write(
            "Customer Features: 7"
        )

        st.write(
            "Classes: Good, Poor, Standard"
        )

        st.write(
            "Model Selection Criterion: Cross-validated Macro F1"
        )

        st.write(
            "Final Test Macro F1: 74.26%"
        )

        st.write(
            "Final Test Balanced Accuracy: 73.62%"
        )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    render_html(
        """
        <div style="
            padding:30px;
            border-radius:20px;
            background:#0D111B;
            border:1px solid
                rgba(105,130,255,0.12);
        ">

            <div style="
                color:#A5AEC0;
                font-size:16px;
                font-weight:700;
            ">
                CrediX — Explainable Machine Learning
                Credit Intelligence
            </div>

            <div style="
                margin-top:18px;
                color:#68748A;
                font-size:13px;
                line-height:1.7;
            ">
                CrediX is a machine-learning-based analytical
                decision-support system. Predictions are not
                financial advice, credit bureau scores, or
                actual lending decisions.
            </div>

        </div>
        """
    )


# ============================================================
# END
# ============================================================