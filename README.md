# Fraud Detection for E-commerce and Banking Transactions

This project focuses on developing robust fraud detection models for e-commerce and banking transactions at Adey Innovations Inc.  By leveraging advanced machine learning techniques, geolocation analysis, transaction pattern recognition, and model explainability tools, we aim to significantly improve fraud detection accuracy and enhance transaction security.

## Overview

Fraud detection is crucial for maintaining trust and preventing financial losses in the financial technology sector. This project addresses the need for accurate and adaptable fraud detection models capable of handling the unique characteristics of both e-commerce and bank credit transaction data.  Effective fraud detection systems empower businesses to identify and respond to fraudulent activities in real-time, minimizing risks and protecting customers.  Furthermore, understanding *why* a model makes a certain prediction is critical for trust and debugging.

## Business Need

Adey Innovations Inc. requires a robust fraud detection solution to minimize financial losses and build customer trust.  This project aims to develop and deploy such a solution, leveraging cutting-edge data science techniques, including model explainability, to identify fraudulent transactions with high accuracy and provide insights into the model's decision-making process.

## Project Goals

* Analyze and preprocess transaction data from various sources.
* Engineer relevant features to enhance fraud pattern identification.
* Develop and train machine learning models for fraud detection.
* Evaluate and optimize model performance.
* Deploy models for real-time fraud detection and establish continuous monitoring.
* Provide model explainability using SHAP and LIME.

## Data and Features

The project utilizes the following datasets:

* **`Fraud_Data.csv`**: E-commerce transaction data.
    * `user_id`: Unique user identifier.
    * `signup_time`: User signup timestamp.
    * `purchase_time`: Purchase timestamp.
    * `purchase_value`: Transaction value.
    * `device_id`: Device identifier.
    * `source`: Traffic source (e.g., SEO, Ads).
    * `browser`: Browser used for transaction.
    * `sex`: User gender.
    * `age`: User age.
    * `ip_address`: Transaction IP address.
    * `class`: Fraudulent transaction indicator (1 = Fraudulent, 0 = Non-Fraudulent).

* **`IpAddress_to_Country.csv`**: IP address to country mapping.
    * `lower_bound_ip_address`: Lower IP address bound.
    * `upper_bound_ip_address`: Upper IP address bound.
    * `country`: Corresponding country.

* **`creditcard.csv`**: Bank credit card transaction data.
    * `Time`: Time elapsed since first transaction.
    * `V1` - `V28`: Anonymized features (PCA transformed).
    * `Amount`: Transaction amount.
    * `Class`: Fraudulent transaction indicator (1 = Fraudulent, 0 = Non-Fraudulent).

## Tasks

### Task 1: Data Analysis and Preprocessing

* **Handle Missing Values**: Imputation or removal.
* **Data Cleaning**: Duplicate removal, data type correction.
* **Exploratory Data Analysis (EDA)**: Univariate and bivariate analysis.
* **Merge Datasets for Geolocation Analysis**: Convert IP addresses to integers and merge `Fraud_Data.csv` with `IpAddress_to_Country.csv`.
* **Feature Engineering**:
    * Transaction frequency and velocity (for `Fraud_Data.csv`).
    * Time-based features (hour of day, day of week) (for `Fraud_Data.csv`).
* **Normalization and Scaling**.
* **Encode Categorical Features**.

### Task 2: Model Building and Training

* **Data Preparation**:
    * Feature and target separation (`Class` for `creditcard.csv`, `class` for `Fraud_Data.csv`).
    * Train-test split.
* **Model Selection**:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Gradient Boosting
    * Multi-Layer Perceptron (MLP)
    * Convolutional Neural Network (CNN)
    * Recurrent Neural Network (RNN)
    * Long Short-Term Memory (LSTM)
* **Model Training and Evaluation**: Train models for both datasets.
* **MLOps Steps**:
    * **Versioning and Experiment Tracking**: Use MLflow to track experiments, log parameters, metrics, and version models.
### Task 3: Model Explainability

Model explainability is crucial for understanding, trust, and debugging in machine learning models. We will use SHAP (Shapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to interpret the models built for fraud detection.

* **Using SHAP for Explainability**:
    SHAP values provide a unified measure of feature importance, explaining the contribution of each feature to the prediction.
    * Installing SHAP: `pip install shap`
    * Explaining a Model with SHAP
    * SHAP Plots:
        * **Summary Plot**: Provides an overview of the most important features.
        * **Force Plot**: Visualizes the contribution of features for a single prediction.
        * **Dependence Plot**: Shows the relationship between a feature and the model output.

* **Using LIME for Explainability**:
    LIME explains individual predictions by approximating the model locally with an interpretable model.
    * Installing LIME: `pip install lime`
    * Explaining a Model with LIME
    * LIME Plots:
        * **Feature Importance Plot**: Shows the most influential features for a specific prediction.
