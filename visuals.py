import matplotlib.pyplot as plt
import numpy as np
import shap

# SAFE IMPORT
try:
    import seaborn as sns
except:
    sns = None

from sklearn.metrics import confusion_matrix


# ---------------- CONFUSION MATRIX ----------------
def plot_confusion_matrix(model, X, y, st):

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(4, 3))  # smaller size

    if sns:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    else:
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                ax.text(j, i, cm[i][j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    st.pyplot(fig)


# ---------------- FEATURE IMPORTANCE ----------------
def plot_feature_importance(model, feature_names, st):

    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(5, 3))  # reduced size

    if sns:
        sns.barplot(x=importances, y=feature_names, ax=ax)
    else:
        ax.barh(feature_names, importances)

    ax.set_title("Feature Importance")

    plt.tight_layout()
    st.pyplot(fig)


# ---------------- PREDICTION PROBABILITY ----------------
def plot_prediction_proba(model, input_df, st):

    probs = model.predict_proba(input_df)[0]
    classes = model.classes_

    fig, ax = plt.subplots(figsize=(4, 3))  # smaller

    ax.bar(classes, probs)

    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    plt.tight_layout()
    st.pyplot(fig)


# ---------------- SHAP SUMMARY ----------------
def plot_shap_summary(model, X, st):

    try:
        explainer = shap.TreeExplainer(model)

        sample_X = X.sample(100)
        shap_values = explainer.shap_values(sample_X)

        fig = plt.figure(figsize=(6, 4))  # controlled size
        shap.summary_plot(shap_values, sample_X, show=False)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.write("SHAP Summary Error:", e)