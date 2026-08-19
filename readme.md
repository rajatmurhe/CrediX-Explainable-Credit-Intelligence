# CrediX - Explainable Credit Intelligence

CrediX is a machine learning based credit risk assessment system that classifies customer profiles into Good, Standard, and Poor credit categories.

The project combines data preprocessing, machine learning model comparison, hyperparameter tuning, probability calibration, model evaluation, feature importance analysis, and an interactive Streamlit dashboard.

The system was trained and evaluated using 31,711 customer records.

## Project Overview

The objective of CrediX is to build a credit risk classification system that not only predicts a customer's credit category but also provides information about the factors influencing the prediction.

The application allows users to enter a customer's personal and financial information and receive:

- Predicted credit category
- Prediction probabilities
- Model confidence
- Feature importance information
- Financial profile summary
- Basic financial recommendations
- Model evaluation information

CrediX is intended as an academic and research project and is not a replacement for professional financial advice or real-world lending decisions.

## Dataset

The dataset contains 31,711 records and 21 original columns.

Target distribution:

| Credit Score | Records |
|--------------|---------:|
| Standard | 19,730 |
| Good | 7,551 |
| Poor | 4,430 |

The final model uses seven customer-facing features:

| Feature | Description |
|---------|-------------|
| Age | Customer age |
| Occupation | Customer occupation |
| Annual_Income | Annual income |
| Num_of_Delayed_Payment | Number of delayed payments |
| Total_EMI_per_month | Total monthly EMI |
| Outstanding_Debt | Outstanding debt |
| Monthly_Balance | Monthly financial balance |

## Machine Learning Approach

The project evaluates multiple classification algorithms before selecting the final model.

| Model | CV Accuracy | CV Macro F1 |
|-------|------------:|------------:|
| Logistic Regression | 34.57% | 33.55% |
| HistGradientBoosting | 65.44% | 61.24% |
| Extra Trees | 76.83% | 71.86% |
| Random Forest | 74.05% | 66.19% |

Extra Trees was selected because it achieved the highest benchmark Macro F1 score.

## Hyperparameter Tuning

The selected Extra Trees model was tuned using cross-validation.

Final parameters:

```text
n_estimators = 150
max_depth = 24
max_features = 0.8
min_samples_split = 5
min_samples_leaf = 1
````

Best tuned cross-validation Macro F1:

```text
0.7410
```

## Probability Calibration

Since the application displays prediction probabilities, different calibration methods were evaluated.

| Method       | Macro F1 | Brier Score | Log Loss |
| ------------ | -------: | ----------: | -------: |
| Uncalibrated |   0.7410 |      0.3448 |   0.6062 |
| Sigmoid      |   0.7090 |      0.3312 |   0.5682 |
| Isotonic     |   0.7258 |      0.3277 |   0.5602 |

Isotonic calibration was selected because it produced the lowest Brier score and Log Loss among the evaluated calibration methods.

## Final Model Performance

The final model achieved the following test-set results:

| Metric            |  Score |
| ----------------- | -----: |
| Accuracy          | 78.62% |
| Balanced Accuracy | 73.62% |
| Macro Precision   | 74.94% |
| Macro Recall      | 73.62% |
| Macro F1          | 74.26% |
| Weighted F1       | 78.52% |

Class-wise performance:

| Class    | Precision | Recall |     F1 |
| -------- | --------: | -----: | -----: |
| Good     |    74.98% | 72.85% | 73.90% |
| Poor     |    67.62% | 63.88% | 65.70% |
| Standard |    82.22% | 84.14% | 83.17% |

## Confusion Matrix

```text
                Predicted
              Good  Poor  Standard

Actual Good    1100    5      405
Actual Poor       7  566      313
Actual Standard 360  266     3321
```

## Explainability

CrediX provides feature importance analysis to show which variables have the greatest influence on the trained model.

The final model's most important features were:

| Feature                | Importance |
| ---------------------- | ---------: |
| Outstanding_Debt       |     0.1998 |
| Age                    |     0.1477 |
| Total_EMI_per_month    |     0.1447 |
| Num_of_Delayed_Payment |     0.1224 |
| Annual_Income          |     0.1122 |
| Monthly_Balance        |     0.0834 |

The current implementation uses the Extra Trees model's native Gini/MDI feature importance.

## Application

The project includes an interactive Streamlit dashboard.

The dashboard provides:

* Personal profile inputs
* Financial profile inputs
* Credit category prediction
* Prediction probability distribution
* Model confidence
* Financial snapshot
* Feature importance analysis
* Model performance information
* Model input inspection
* Financial recommendations

Users can enter values such as age, occupation, annual income, delayed payments, monthly EMI, outstanding debt, and monthly balance.

The application then passes these values through the same preprocessing and trained model pipeline used during development.

## Project Structure

```text
xai_credit_project/
|
├── app.py
├── model.py
├── report.py
├── pdf_report.py
├── visuals.py
├── test_streamlit.py
|
│
│── credit_data.csv
|
├── trained_credit_model.joblib
├── preprocessor.joblib
├── label_encoder.joblib
├── feature_names.joblib
├── model_metrics.joblib
├── feature_bounds.joblib
|
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

Python

Pandas

NumPy

Scikit-learn

Joblib

Streamlit

Matplotlib

Plotly

ReportLab

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd xai_credit_project
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in the browser.

## Training the Model

The complete model training and evaluation pipeline can be executed using:

```bash
python model.py
```

The script performs:

1. Dataset ingestion
2. Data validation
3. Data sanitization
4. Target distribution analysis
5. Model benchmarking
6. Model selection
7. Hyperparameter tuning
8. Probability calibration
9. Final test evaluation
10. Feature importance analysis
11. Model artifact generation
12. Sanity testing

## Model Files

The training process generates the following files:

```text
trained_credit_model.joblib
preprocessor.joblib
label_encoder.joblib
feature_names.joblib
model_metrics.joblib
feature_bounds.joblib
```

These files are used by the Streamlit application to load the trained model and make predictions without retraining the model.

## Project Workflow

```text
Dataset
   |
   v
Data Cleaning
   |
   v
Feature Preparation
   |
   v
Model Comparison
   |
   v
Extra Trees Selection
   |
   v
Hyperparameter Tuning
   |
   v
Probability Calibration
   |
   v
Model Evaluation
   |
   v
Feature Importance
   |
   v
Streamlit Application
```

## Key Results

The completed system provides an end-to-end machine learning workflow for credit risk classification.

The final model achieved:

* 78.62% test accuracy
* 73.62% balanced accuracy
* 74.26% Macro F1
* 78.52% weighted F1

The application also provides probability estimates and feature importance information to make the model output easier to interpret.

## Limitations

The model is trained on the available dataset and its performance depends on the quality and distribution of that data.

Feature importance describes the influence of features within the trained model and should not be interpreted as proof of causation.

The system should not be used to make real-world lending decisions without additional validation, fairness analysis, regulatory review, and domain-specific testing.

## Future Improvements

Possible extensions include:

* SHAP based local explanations
* LIME explanations
* Fairness and bias analysis
* Model drift detection
* Automated model monitoring
* FastAPI deployment
* Docker deployment
* Cloud deployment
* Model versioning
* Automated retraining

## Author

Rajat Murhe
