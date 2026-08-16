import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():

    np.random.seed(42)

    data = pd.DataFrame({
        "Age": np.random.randint(21, 60, 300),
        "Income": np.random.randint(20000, 100000, 300),
        "Num_Loans": np.random.randint(0, 10, 300),
        "EMI": np.random.randint(1000, 20000, 300),
        "Outstanding_Debt": np.random.randint(0, 50000, 300),
        "Balance": np.random.randint(0, 100000, 300)
    })

    data["Credit_Score"] = np.where(
        (
            (data["Income"] > 50000)
            &
            (data["Outstanding_Debt"] < 20000)
            &
            (data["Num_Loans"] <= 4)
        ),
        "Good",
        np.where(
            (
                (data["Outstanding_Debt"] > 40000)
                |
                (data["Num_Loans"] >= 7)
            ),
            "Poor",
            "Standard"
        )
    )

    return data


# =========================================================
# PREPROCESS DATA
# =========================================================

def preprocess_data(data):

    X = data.drop(
        "Credit_Score",
        axis=1
    )

    y = data["Credit_Score"]

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model():

    data = load_data()

    X, y, label_encoder = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
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
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")

    print()
    print("Classification Report")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    print("Confusion Matrix")
    print("-" * 60)

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("=" * 60)
    print()

    return (
        model,
        X.columns.tolist(),
        label_encoder
    )


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict(model, input_df):

    expected_columns = [
        "Age",
        "Income",
        "Num_Loans",
        "EMI",
        "Outstanding_Debt",
        "Balance"
    ]

    input_df = input_df[
        expected_columns
    ].copy()

    input_df["Age"] = pd.to_numeric(
        input_df["Age"]
    )

    input_df["Income"] = pd.to_numeric(
        input_df["Income"]
    )

    input_df["Num_Loans"] = pd.to_numeric(
        input_df["Num_Loans"]
    )

    input_df["EMI"] = pd.to_numeric(
        input_df["EMI"]
    )

    input_df["Outstanding_Debt"] = pd.to_numeric(
        input_df["Outstanding_Debt"]
    )

    input_df["Balance"] = pd.to_numeric(
        input_df["Balance"]
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


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    model, feature_names, label_encoder = train_model()

    sample_customer = pd.DataFrame([{
        "Age": 30,
        "Income": 60000,
        "Num_Loans": 2,
        "EMI": 5000,
        "Outstanding_Debt": 15000,
        "Balance": 40000
    }])

    prediction, probabilities = predict(
        model,
        sample_customer
    )

    predicted_class = label_encoder.inverse_transform(
        prediction
    )[0]

    print(
        "Predicted Credit Category:",
        predicted_class
    )

    print("Prediction Probabilities:")

    for class_name, probability in zip(
        label_encoder.classes_,
        probabilities[0]
    ):
        print(
            f"{class_name}: {probability * 100:.2f}%"
        )
