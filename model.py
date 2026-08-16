import os
import warnings
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    LabelEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/credit_data.csv"

MODEL_PATH = "trained_credit_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
FEATURE_PATH = "feature_names.joblib"
METRICS_PATH = "model_metrics.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

RANDOM_STATE = 42


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NUMERIC_FEATURES = [
    "Age",
    "Annual_Income",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_History_Age_Months"
]

CATEGORICAL_FEATURES = [
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "Credit_Score"


# ============================================================
# LOAD REAL DATASET
# ============================================================

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 70)
    print("REAL CREDIT DATASET")
    print("=" * 70)

    print(f"Dataset path : {DATA_PATH}")
    print(f"Rows         : {len(df)}")
    print(f"Columns      : {len(df.columns)}")

    # Keep only required columns
    missing_columns = [
        col for col in FEATURES + [TARGET]
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    df = df[FEATURES + [TARGET]].copy()

    # Remove accidental missing target rows
    df = df.dropna(subset=[TARGET])

    print("\nCredit Score Distribution")
    print("-" * 70)
    print(df[TARGET].value_counts())

    print("\nFeatures used:")
    print(FEATURES)

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    df = df.copy()

    # Convert numeric columns safely
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Convert categorical values to strings
    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

    # Replace problematic infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    return df


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(X_train, y_train, preprocessor):

    print()
    print("=" * 70)
    print("5-FOLD CROSS-VALIDATION MODEL COMPARISON")
    print("=" * 70)

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=16,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=150,
                max_depth=16,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    results = {}

    for name, estimator in models.items():

        print()
        print(f"Evaluating: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator)
            ]
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        results[name] = {
            "scores": scores,
            "mean": float(scores.mean()),
            "std": float(scores.std())
        }

        print(
            "Fold Scores :",
            np.round(scores, 4)
        )

        print(
            f"Mean Accuracy : {scores.mean():.4f}"
        )

        print(
            f"Std Deviation : {scores.std():.4f}"
        )

    return results


# ============================================================
# RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

def tune_random_forest(X_train, y_train):

    print()
    print("=" * 70)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 70)

    preprocessor = create_preprocessor()

    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                rf
            )
        ]
    )

    param_grid = {

        "model__n_estimators": [
            150,
            250
        ],

        "model__max_depth": [
            12,
            16,
            None
        ],

        "model__min_samples_split": [
            2,
            5
        ],

        "model__min_samples_leaf": [
            1
        ]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print()
    print("Best Parameters:")
    print(grid_search.best_params_)

    print(
        f"Best CV Accuracy: "
        f"{grid_search.best_score_:.4f}"
    )

    return grid_search


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

def train_model():

    df = load_data()

    df = clean_data(df)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print()
    print("Encoded classes:")
    print(label_encoder.classes_)

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Testing samples  : {len(X_test)}"
    )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison_results = compare_models(
        X_train,
        y_train,
        create_preprocessor()
    )

    # --------------------------------------------------------
    # HYPERPARAMETER TUNING
    # --------------------------------------------------------

    search = tune_random_forest(
        X_train,
        y_train
    )

    final_model = search.best_estimator_

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    y_pred = final_model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print()
    print("=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print()
    print("Classification Report")
    print("-" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("-" * 70)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    rf_model = final_model.named_steps["model"]

    preprocessor = (
        final_model
        .named_steps["preprocessor"]
    )

    transformed_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({

        "Feature":
            transformed_names,

        "Importance":
            importances
    })

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 70)
    print("TOP FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        feature_importance_df.head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1": float(f1),

        "confusion_matrix": cm.tolist(),

        "classification_report":
            classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
                zero_division=0
            ),

        "comparison": comparison_results,

        "best_params":
            search.best_params_,

        "cv_accuracy":
            float(search.best_score_),

        "test_samples":
            int(len(X_test)),

        "train_samples":
            int(len(X_train))
    }

    # --------------------------------------------------------
    # SAVE ARTIFACTS
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_PATH
    )

    joblib.dump(
        label_encoder,
        ENCODER_PATH
    )

    joblib.dump(
        FEATURES,
        FEATURE_PATH
    )

    joblib.dump(
        metrics,
        METRICS_PATH
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"Model       : {MODEL_PATH}"
    )

    print(
        f"Encoder     : {ENCODER_PATH}"
    )

    print(
        f"Features    : {FEATURE_PATH}"
    )

    print(
        f"Metrics     : {METRICS_PATH}"
    )

    print(
        f"Preprocessor: {PREPROCESSOR_PATH}"
    )

    print("=" * 70)

    return (
        final_model,
        FEATURES,
        label_encoder
    )


# ============================================================
# PREDICTION
# ============================================================

def predict(model, input_df):

    input_df = input_df.copy()

    # Make sure every expected feature exists
    for feature in FEATURES:

        if feature not in input_df.columns:
            input_df[feature] = np.nan

    input_df = input_df[FEATURES]

    for col in NUMERIC_FEATURES:

        input_df[col] = pd.to_numeric(
            input_df[col],
            errors="coerce"
        )

    for col in CATEGORICAL_FEATURES:

        input_df[col] = (
            input_df[col]
            .astype(str)
        )

    prediction = model.predict(
        input_df
    )

    probabilities = model.predict_proba(
        input_df
    )

    return (
        prediction,
        probabilities
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    model, feature_names, label_encoder = train_model()

    sample_customer = pd.DataFrame([{

        "Age": 30,

        "Occupation": "Engineer",

        "Annual_Income": 60000,

        "Num_Bank_Accounts": 3,

        "Num_Credit_Card": 4,

        "Interest_Rate": 8,

        "Num_of_Loan": 2,

        "Delay_from_due_date": 5,

        "Num_of_Delayed_Payment": 1,

        "Changed_Credit_Limit": 10,

        "Num_Credit_Inquiries": 2,

        "Credit_Mix": "Good",

        "Outstanding_Debt": 15000,

        "Credit_Utilization_Ratio": 25,

        "Payment_of_Min_Amount": "Yes",

        "Total_EMI_per_month": 5000,

        "Amount_invested_monthly": 5000,

        "Payment_Behaviour":
            "High_spent_Medium_value_payments",

        "Monthly_Balance": 40000,

        "Credit_History_Age_Months": 250

    }])

    prediction, probabilities = predict(
        model,
        sample_customer
    )

    predicted_class = (
        label_encoder
        .inverse_transform(prediction)[0]
    )

    print()
    print("=" * 70)
    print("EXAMPLE CUSTOMER PREDICTION")
    print("=" * 70)

    print(
        "Predicted Credit Category:",
        predicted_class
    )

    print()
    print("Prediction Probabilities:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):

        print(
            f"{class_name}: "
            f"{probability * 100:.2f}%"
        )