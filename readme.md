# CrediX — Explainable Machine Learning Credit Intelligence

CrediX is an Explainable Machine Learning-based credit assessment platform that analyses a user's financial profile, predicts their credit category, explains the factors influencing the prediction, and provides personalised financial improvement recommendations.

The system combines machine learning prediction, model explainability, visual analytics, and automated PDF reporting into a customer-focused web application.

---

## Project Overview

Traditional credit assessment systems often provide a prediction or score without clearly explaining why a particular decision was made.

CrediX addresses this problem by combining:

- Machine Learning-based credit classification
- Random Forest ensemble learning
- Explainable AI using SHAP
- Feature importance analysis
- Prediction confidence visualization
- Financial profile analysis
- Personalised recommendations
- Automated PDF report generation
- Interactive Streamlit dashboard

The objective is not only to predict a credit category but also to make the prediction understandable to the user.

---

## Key Features

### 1. Credit Category Prediction

The system analyses financial information provided by the user and predicts their credit category using a trained Random Forest classifier.

The application presents:

- Predicted credit category
- Model confidence
- Model used for prediction
- Financial profile summary

---

### 2. Explainable AI

CrediX uses SHAP (SHapley Additive exPlanations) to explain how individual financial features influence the model prediction.

This helps answer questions such as:

- Which financial factor influenced the prediction the most?
- Did income support the prediction?
- Did outstanding debt negatively influence the prediction?
- Which features should the user improve?

---

### 3. Personalised Recommendations

The system generates recommendations based on the user's financial profile.

Examples include:

- Reducing outstanding debt
- Managing loan exposure
- Reducing monthly EMI burden
- Building a stronger financial reserve
- Maintaining healthy financial behaviour

These recommendations are generated dynamically based on the entered financial information.

---

### 4. Prediction Confidence

The application displays the probability assigned by the machine learning model to each possible credit category.

This allows users to understand not only the predicted category but also how confident the model is in its prediction.

---

### 5. Financial Snapshot

The dashboard provides a quick summary of important financial indicators:

- Annual income
- Outstanding debt
- Monthly EMI
- Number of active loans
- Account balance

---

### 6. Interactive Dashboard

The application is implemented using Streamlit and provides an interactive customer-facing dashboard.

The interface contains:

- Credit profile input section
- Prediction results
- Confidence analysis
- Recommendations
- Financial metrics
- Explainable AI section
- Model performance information
- PDF report generation

---

### 7. PDF Credit Report

CrediX provides an automated PDF reporting feature.

The report can contain:

- Financial profile
- Predicted credit category
- Prediction probabilities
- Personalised recommendations
- Credit assessment information

This allows the assessment to be saved and shared as a structured report.

---

# Machine Learning Pipeline

The overall system follows the following workflow:

```text
User Financial Data
        |
        v
Data Preprocessing
        |
        v
Feature Preparation
        |
        v
Machine Learning Model
        |
        v
Random Forest Classifier
        |
        +--------------------+
        |                    |
        v                    v
Credit Prediction       Prediction Probability
        |
        v
Explainable AI
        |
        v
SHAP Feature Analysis
        |
        v
Personalised Recommendations
        |
        v
Interactive Dashboard
        |
        v
PDF Credit Report
