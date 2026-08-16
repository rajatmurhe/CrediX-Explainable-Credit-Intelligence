# CrediX — Explainable Machine Learning Credit Intelligence

CrediX is an Explainable Machine Learning-based credit assessment platform that analyses a user's financial profile, predicts their credit category, explains the factors influencing the prediction, and provides personalised financial recommendations.

The system combines Machine Learning, Explainable AI, financial analytics, interactive visualization, and automated PDF reporting into a customer-focused credit intelligence platform.

---

## Project Overview

Traditional credit assessment systems often provide a credit score or category without clearly explaining why a particular prediction was made.

CrediX addresses this problem by combining machine learning prediction with Explainable AI.

The system analyses a user's financial and credit-related information and predicts one of three credit categories:

- Good
- Standard
- Poor

Instead of providing only a prediction, CrediX also provides:

- Prediction confidence
- Feature importance
- SHAP-based explanations
- Financial profile analysis
- Personalised recommendations
- Interactive visualizations
- Model performance analysis
- Automated PDF credit reports

The goal is to make machine learning-based credit assessment more transparent, interpretable, and user-friendly.

---

# Problem Statement

Credit assessment is an important financial decision-making process. Machine learning models can identify complex relationships within financial data, but many models provide predictions without clearly explaining the reasoning behind them.

This creates several challenges:

- Lack of transparency
- Difficulty interpreting model predictions
- Limited understanding of influential financial factors
- Difficulty identifying areas for financial improvement
- Black-box decision making

CrediX attempts to address these challenges by combining:

```text
Machine Learning
        +
Explainable AI
        +
Financial Analytics
        +
Personalised Recommendations
        +
Interactive Dashboard
        +
Automated Reporting
````

---

# Objectives

The main objectives of CrediX are:

1. Develop a Machine Learning-based credit classification system.
2. Use a real credit dataset instead of artificially generated data.
3. Compare multiple classification algorithms.
4. Perform stratified train/test evaluation.
5. Perform 5-fold cross-validation.
6. Perform Random Forest hyperparameter tuning.
7. Evaluate the final model using multiple performance metrics.
8. Explain model predictions using SHAP.
9. Identify important financial and credit-related features.
10. Generate personalised financial recommendations.
11. Provide an interactive customer-facing dashboard.
12. Generate downloadable PDF credit assessment reports.

---

# Dataset

CrediX uses a real credit dataset containing:

```text
Records: 31,711
Columns: 21
Target Classes: 3
```

The target variable is:

```text
Credit_Score
```

The three classes are:

```text
Good
Poor
Standard
```

## Class Distribution

| Credit Category |    Samples |
| --------------- | ---------: |
| Standard        |     19,730 |
| Good            |      7,551 |
| Poor            |      4,430 |
| **Total**       | **31,711** |

The dataset contains both numerical and categorical financial features.

---

# Features Used

The final model uses 20 predictive features.

## Numerical Features

```text
Age
Annual_Income
Num_Bank_Accounts
Num_Credit_Card
Interest_Rate
Num_of_Loan
Delay_from_due_date
Num_of_Delayed_Payment
Changed_Credit_Limit
Num_Credit_Inquiries
Outstanding_Debt
Credit_Utilization_Ratio
Total_EMI_per_month
Amount_invested_monthly
Monthly_Balance
Credit_History_Age_Months
```

## Categorical Features

```text
Occupation
Credit_Mix
Payment_of_Min_Amount
Payment_Behaviour
```

---

# Machine Learning Pipeline

The complete CrediX Machine Learning pipeline is:

```text
Real Credit Dataset
        |
        v
Data Cleaning
        |
        v
Feature Selection
        |
        v
Numerical + Categorical Preprocessing
        |
        v
Stratified Train/Test Split
        |
        v
5-Fold Cross Validation
        |
        v
Model Comparison
        |
        +-------------------+
        |                   |
        v                   v
Logistic Regression    Decision Tree
        |                   |
        +---------+---------+
                  |
                  v
            Random Forest
                  |
                  v
       Hyperparameter Tuning
                  |
                  v
          Final Random Forest
                  |
                  v
        Model Evaluation
                  |
          +-------+-------+
          |       |       |
          v       v       v
       SHAP   Feature   Prediction
              Importance Probability
                  |
                  v
        Personalised Recommendations
                  |
                  v
          Streamlit Dashboard
                  |
                  v
             PDF Report
```

---

# Data Preprocessing

The dataset contains both numerical and categorical variables.

## Numerical Processing

Numerical features are processed using:

* Missing-value imputation
* Numeric conversion
* Infinite-value handling
* Standardization

The numerical preprocessing pipeline uses:

```text
SimpleImputer(strategy="median")
        ↓
StandardScaler()
```

## Categorical Processing

Categorical features are processed using:

```text
SimpleImputer(strategy="most_frequent")
        ↓
OneHotEncoder(handle_unknown="ignore")
```

This converts categorical financial information into machine-learning-compatible numerical representations.

---

# Train/Test Split

The dataset is divided using a stratified train/test split.

```text
Training Samples : 25,368
Testing Samples  : 6,343
```

The stratification ensures that the distribution of the three credit classes is preserved between training and testing data.

---

# Machine Learning Models

CrediX evaluates three different classification algorithms.

## 1. Logistic Regression

Logistic Regression is used as a linear baseline model.

It provides a comparison between a linear classification approach and nonlinear tree-based approaches.

---

## 2. Decision Tree

Decision Tree is used as a tree-based baseline classifier.

It recursively splits the data based on feature values to create classification decisions.

Decision Trees are relatively easy to interpret and can capture nonlinear relationships.

---

## 3. Random Forest

Random Forest is the primary model used by the final CrediX system.

Random Forest is an ensemble learning algorithm that combines multiple decision trees to produce a more robust prediction.

Advantages include:

* Handles nonlinear relationships
* Works well with structured data
* Captures feature interactions
* Provides feature importance
* Works effectively with SHAP
* Reduces reliance on a single decision tree

---

# Model Comparison

5-fold stratified cross-validation was used to compare the three models.

| Model               | Mean CV Accuracy |
| ------------------- | ---------------: |
| Logistic Regression |       **68.02%** |
| Decision Tree       |       **69.43%** |
| Random Forest       |       **74.37%** |

Random Forest achieved the highest cross-validation performance before hyperparameter tuning.

---

# Hyperparameter Tuning

Random Forest hyperparameters were optimized using `GridSearchCV` with 5-fold cross-validation.

The search evaluated different combinations of:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

The best configuration was:

```text
n_estimators       = 250
max_depth          = None
min_samples_split  = 2
min_samples_leaf   = 1
class_weight       = balanced
max_features       = sqrt
random_state       = 42
```

The tuned Random Forest achieved:

```text
5-Fold Cross-Validation Accuracy = 78.66%
```

---

# Final Model Performance

The final Random Forest model was evaluated on the untouched test dataset containing:

```text
6,343 samples
```

The final results were:

| Metric        |      Score |
| ------------- | ---------: |
| **Accuracy**  | **80.20%** |
| **Precision** | **79.86%** |
| **Recall**    | **80.20%** |
| **F1-Score**  | **79.65%** |

---

# Classification Report

The final model achieved:

| Credit Category      | Precision |   Recall | F1-Score |   Support |
| -------------------- | --------: | -------: | -------: | --------: |
| Good                 |      0.76 |     0.73 |     0.75 |     1,510 |
| Poor                 |      0.76 |     0.53 |     0.62 |       886 |
| Standard             |      0.82 |     0.89 |     0.85 |     3,947 |
| **Weighted Average** |  **0.80** | **0.80** | **0.80** | **6,343** |

The model performs particularly well on the Standard category, while the Poor category remains more challenging to classify.

---

# Confusion Matrix

The final model produced the following confusion matrix:

```text
                  Predicted
              Good   Poor   Standard

Good          1106     21       383

Poor            41    467       378

Standard       303    130      3514
```

The model correctly classified:

```text
Good:
1106 / 1510

Poor:
467 / 886

Standard:
3514 / 3947
```

The Standard category has the highest recall, while Poor is the most difficult category for the current model.

---

# Feature Importance

The Random Forest model provides feature importance values.

The most influential features in the final model were:

| Rank | Feature                   | Importance |
| ---: | ------------------------- | ---------: |
|    1 | Delay_from_due_date       |     0.0781 |
|    2 | Outstanding_Debt          |     0.0630 |
|    3 | Interest_Rate             |     0.0620 |
|    4 | Num_Credit_Card           |     0.0559 |
|    5 | Credit_Mix_Good           |     0.0520 |
|    6 | Credit_History_Age_Months |     0.0514 |
|    7 | Credit_Mix_Standard       |     0.0506 |
|    8 | Annual_Income             |     0.0504 |
|    9 | Monthly_Balance           |     0.0482 |
|   10 | Changed_Credit_Limit      |     0.0468 |

This provides a global understanding of which variables are most influential within the Random Forest model.

---

# Explainable AI

CrediX uses SHAP:

```text
SHapley Additive exPlanations
```

to explain machine learning predictions.

SHAP provides feature-level contribution values that help explain how individual variables influence a prediction.

The explainability pipeline is:

```text
User Financial Profile
        |
        v
Trained Random Forest
        |
        v
Prediction
        |
        v
SHAP Explainer
        |
        v
Feature Contributions
        |
        v
Human-Readable Explanation
```

---

# Why SHAP?

A prediction alone does not explain the reasoning behind a machine learning decision.

For example:

```text
Predicted Credit Category:
Standard
```

does not explain why the model reached that decision.

SHAP allows CrediX to investigate the contribution of individual financial factors.

This can help answer questions such as:

* Which financial factors influenced the prediction?
* Which factors increased or decreased the model output?
* Which variables were most influential for the individual prediction?

---

# Prediction Confidence

CrediX uses the probability output of the Random Forest classifier to display the model's confidence across the three credit categories.

Example:

```text
Good       28.8%
Poor       21.2%
Standard   50.0%
```

The category with the highest predicted probability becomes the final predicted class.

---

# Personalised Recommendations

The platform provides financial recommendations based on the user's financial profile.

Examples include:

```text
Reduce outstanding debt

Manage loan exposure

Reduce excessive EMI burden

Maintain a healthy account balance

Improve payment behaviour

Monitor credit utilization
```

These recommendations are intended as analytical suggestions and are not professional financial advice.

---

# Interactive Dashboard

The CrediX dashboard is developed using Streamlit.

The dashboard provides an interactive interface for:

* Entering financial information
* Running credit predictions
* Viewing model confidence
* Viewing financial metrics
* Viewing recommendations
* Viewing feature importance
* Viewing SHAP explanations
* Viewing model performance
* Generating PDF reports

---

# Dashboard Workflow

```text
Enter Financial Information
             |
             v
Analyse Credit Profile
             |
             v
Machine Learning Prediction
             |
             v
Credit Category
             |
             v
Prediction Confidence
             |
             v
Financial Snapshot
             |
             v
Personalised Recommendations
             |
             v
Explainable AI
             |
             v
Feature Importance
             |
             v
PDF Credit Report
```

---

# PDF Report Generation

CrediX provides automated PDF report generation using ReportLab.

The generated report can include:

* Financial profile
* Predicted credit category
* Prediction probabilities
* Model information
* Financial recommendations
* Credit assessment information

This provides a structured and shareable version of the assessment.

---

# Technology Stack

## Programming Language

* Python

## Machine Learning

* Scikit-learn
* Random Forest
* Decision Tree
* Logistic Regression

## Data Processing

* Pandas
* NumPy

## Explainable AI

* SHAP

## Visualization

* Matplotlib
* Seaborn

## Web Application

* Streamlit

## PDF Generation

* ReportLab

## Model Persistence

* Joblib

## Development Tools

* Visual Studio Code
* Git
* GitHub
* Python Virtual Environment

---

# Project Structure

```text
CrediX/
│
├── app.py
├── model.py
├── visuals.py
├── pdf_report.py
├── report.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── credit_data.csv
│
├── trained_credit_model.joblib
├── label_encoder.joblib
├── feature_names.joblib
├── model_metrics.joblib
├── preprocessor.joblib
│
├── screenshots/
│   ├── dashboard.png
│   ├── credit_assessment.png
│   ├── recommendations.png
│   ├── prediction_confidence.png
│   ├── feature_importance.png
│   ├── shap_explanation.png
│   └── pdf_report.png
│
└── outputs/
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── prediction_confidence.png
```

---

# Installation

## Prerequisites

Install the following before running CrediX:

* Python 3.11+
* pip
* Git

---

# Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/CrediX.git
```

Navigate to the project:

```bash
cd CrediX
```

---

# Create a Virtual Environment

## macOS / Linux

```bash
python3 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

## Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

# Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
pandas
numpy
scikit-learn
shap
matplotlib
seaborn
streamlit
reportlab
joblib
```

---

# Training the Model

The model can be trained using:

```bash
python model.py
```

The training pipeline performs:

```text
Data Loading
     ↓
Data Cleaning
     ↓
Feature Preprocessing
     ↓
Train/Test Split
     ↓
5-Fold Cross Validation
     ↓
Model Comparison
     ↓
Random Forest Hyperparameter Tuning
     ↓
Final Model Training
     ↓
Model Evaluation
     ↓
Model Persistence
```

---

# Saved Model Artifacts

After successful training, the following files are generated:

```text
trained_credit_model.joblib
label_encoder.joblib
feature_names.joblib
model_metrics.joblib
preprocessor.joblib
```

These files store the trained machine learning model, label encoding information, feature information, evaluation metrics, and preprocessing pipeline.

---

# Run the Streamlit Application

After installing dependencies:

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

Open the URL in a browser.

---

# Usage

## Step 1 — Enter Financial Information

Enter the required financial and credit information into the dashboard.

---

## Step 2 — Analyse Credit Profile

Click the credit analysis button.

The application sends the information through the trained machine learning pipeline.

---

## Step 3 — View Prediction

The dashboard displays:

* Predicted credit category
* Prediction probability
* Model confidence

---

## Step 4 — View Recommendations

The system provides personalised financial recommendations based on the financial profile.

---

## Step 5 — View Explainable AI

The user can inspect the factors contributing to the prediction through SHAP-based explanations.

---

## Step 6 — View Visual Analytics

The dashboard can display:

* Feature importance
* Prediction confidence
* Confusion matrix
* SHAP analysis
* Model performance

---

## Step 7 — Generate PDF

The user can generate a structured PDF credit assessment report.

---

# Screenshots

Add screenshots of the actual working application to the `screenshots/` directory.

## Main Dashboard

```markdown
![CrediX Dashboard](screenshots/dashboard.png)
```

## Credit Assessment

```markdown
![Credit Assessment](screenshots/credit_assessment.png)
```

## Recommendations

```markdown
![Personalised Recommendations](screenshots/recommendations.png)
```

## Prediction Confidence

```markdown
![Prediction Confidence](screenshots/prediction_confidence.png)
```

## Feature Importance

```markdown
![Feature Importance](screenshots/feature_importance.png)
```

## SHAP Explanation

```markdown
![SHAP Explanation](screenshots/shap_explanation.png)
```

## PDF Report

```markdown
![PDF Report](screenshots/pdf_report.png)
```

---

# Outputs

CrediX produces several categories of outputs.

## Machine Learning Outputs

```text
Credit Category
Prediction Probability
Accuracy
Precision
Recall
F1-Score
Classification Report
Confusion Matrix
```

## Explainability Outputs

```text
Feature Importance
SHAP Feature Contributions
Individual Prediction Explanation
```

## Dashboard Outputs

```text
Credit Assessment
Financial Snapshot
Prediction Confidence
Personalised Recommendations
Visual Analytics
```

## Reporting Outputs

```text
Automated PDF Credit Report
```

---

# Advantages

CrediX provides:

1. Real-dataset-based machine learning.
2. 31,711 financial records for model development.
3. 20 predictive features.
4. Multiple machine learning models.
5. 5-fold cross-validation.
6. Hyperparameter tuning.
7. 80.20% held-out test accuracy.
8. SHAP-based explainability.
9. Feature importance analysis.
10. Prediction probability analysis.
11. Personalised recommendations.
12. Interactive Streamlit dashboard.
13. Automated PDF reporting.
14. Persisted trained model artifacts.

---

# Limitations

Although the model demonstrates useful performance, several limitations remain.

## 1. Dataset Limitations

The model is dependent on the characteristics and quality of the available dataset.

## 2. Class Performance

The model performs better on the Standard class than the Poor class.

The Poor class achieved:

```text
Precision: 0.76
Recall:    0.53
F1-Score:  0.62
```

Further work is required to improve minority-class classification.

## 3. Production Deployment

The current implementation is an academic and portfolio prototype rather than a production banking credit-scoring system.

## 4. Financial Decisions

The predictions and recommendations should not be considered professional financial advice.

## 5. Generalization

Additional external datasets would be required to establish how well the model generalizes to other populations and financial environments.

---

# Future Enhancements

## 1. Advanced Machine Learning

Future versions could evaluate:

* XGBoost
* LightGBM
* Gradient Boosting
* Support Vector Machines
* Neural Networks

---

## 2. Improved Class Imbalance Handling

Future work could investigate:

* SMOTE
* Borderline-SMOTE
* Cost-sensitive learning
* Threshold optimization
* Balanced ensemble methods

The objective would be to improve performance on the Poor class without unnecessarily reducing performance on the other classes.

---

## 3. Advanced Explainability

Future versions could include:

* LIME
* Counterfactual explanations
* Interactive SHAP plots
* What-if analysis

---

## 4. What-If Credit Simulation

Users could modify financial variables and immediately see how the model's prediction changes.

Example:

```text
Current Profile
       |
       v
Predicted Category
       |
       v
What-If Analysis
       |
       +---- Reduce Outstanding Debt
       |
       +---- Reduce EMI
       |
       +---- Improve Payment Behaviour
       |
       +---- Increase Balance
       |
       v
New Model Prediction
```

---

## 5. Model Monitoring

A production version could monitor:

* Data drift
* Prediction drift
* Feature distribution
* Model performance
* Class distribution changes

---

## 6. Cloud Deployment

The application could be deployed using:

* Streamlit Community Cloud
* Docker
* AWS
* Microsoft Azure
* Google Cloud

---

## 7. Security Enhancements

A production implementation could include:

* User authentication
* Role-based access control
* Encryption
* Secure API communication
* Secure database storage
* Audit logging

---

# Ethical Considerations

Credit assessment systems can influence important financial decisions.

A production implementation should therefore consider:

* Data privacy
* Fairness
* Bias detection
* Explainability
* Transparency
* Regulatory compliance
* Human oversight

CrediX is designed as an academic and educational decision-support prototype and should not be used as an autonomous financial decision-making system.

---

# Technical Contribution

The technical contribution of CrediX is the integration of multiple components into a single Explainable Machine Learning workflow:

```text
Real Credit Dataset
        +
Data Preprocessing
        +
Multiple ML Models
        +
Cross Validation
        +
Hyperparameter Tuning
        +
Random Forest
        +
SHAP Explainability
        +
Feature Importance
        +
Prediction Confidence
        +
Personalised Recommendations
        +
Streamlit Dashboard
        +
PDF Reporting
```

The project demonstrates an end-to-end Machine Learning workflow rather than simply training a classifier.

---

# Final Model Summary

The final CrediX model uses:

```text
Dataset:
31,711 records

Predictive Features:
20

Target Classes:
3

Models Compared:
3

Cross Validation:
5-Fold Stratified CV

Best Model:
Tuned Random Forest

Best CV Accuracy:
78.66%

Final Test Samples:
6,343

Final Test Accuracy:
80.20%

Precision:
79.86%

Recall:
80.20%

F1-Score:
79.65%
```

---

# Conclusion

CrediX demonstrates how Machine Learning and Explainable AI can be combined to create a transparent credit assessment platform.

Rather than producing only a credit classification, the system attempts to provide a complete decision-support workflow:

```text
What is the predicted credit category?

How confident is the model?

Which financial factors influenced the prediction?

What areas of the financial profile could potentially be improved?
```

The final system uses a real dataset containing 31,711 records and 20 predictive features. Three machine learning algorithms were compared using 5-fold cross-validation, followed by Random Forest hyperparameter tuning.

The final tuned Random Forest achieved:

```text
5-Fold CV Accuracy : 78.66%

Test Accuracy      : 80.20%
Precision           : 79.86%
Recall              : 80.20%
F1-Score            : 79.65%
```

The project combines these machine learning capabilities with SHAP-based explainability, feature importance analysis, personalised recommendations, interactive visualization, and automated PDF reporting.

---

# Author

**Rajat Murhe**

B.Tech Computer Science Engineering
Specialization: Artificial Intelligence & Machine Learning

---

# License

This project was developed for academic, educational, and portfolio purposes.

````

