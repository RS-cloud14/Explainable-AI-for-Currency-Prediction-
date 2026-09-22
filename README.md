# Explainable AI (XAI) for Daily FX Return Forecasting: GBP/AUD

An end-to-end quantitative pipeline using ensemble machine learning techniques (Random Forest and Bagged Decision Trees) combined with Explainable AI (SHAP and LIME) to forecast daily percentage returns for the British Pound / Australian Dollar (GBP/AUD) currency pair.

# Project Overview

Foreign exchange markets are governed by complex macroeconomic variables, commodity price dynamics, and technical momentum. This project evaluates whether non-linear ensemble models can forecast short-term GBP/AUD daily returns using technical lags, rolling momentum, commodity shocks, and monetary policy indicators.

# Key Objectives

Ingest and align macroeconomic, central bank, and commodity time series (2021–2024).

Prune collinear features using iterative Variance Inflation Factor (VIF) screening.

Contrast an unconstrained Random Forest against an ensemble of deep Bagged Decision Trees under temporal cross-validation.

Interpret both global feature interactions (SHAP summary and permutation importances) and local instance predictions (LIME).

# Dataset Summary

Currency Pair: GBP/AUD

Period: January 1, 2021 to June 30, 2024 (1,277 initial daily records)

Target Variable: GBPAAUD_Daily_Return (daily percentage change)

External Macro Indicators: Gold Price Return, Crude Oil Return, Global Economic Policy Uncertainty (GEPU), UK Index, UK Interest Rate, AU Interest Rate, UK Inflation, AU Inflation

Technical Indicators: 1-day to 3-day lagged returns, rolling moving averages (3, 5, 10 days), and rolling standard deviations (3, 5, 10 days)

# Pipeline Architecture

Data Ingestion & Cleaning: Date filtering (2021-01-01 to 2024-06-30), sorting, and null handling.

Feature Engineering: Computing percentage returns, rolling mean/volatility features, and lagged returns.

Temporal Train/Test Split: 80% chronologically ordered training set, 20% holdout test set (no shuffling).

Time Series Cross-Validation: 5-fold TimeSeriesSplit on the training set to prevent temporal leakage.

Multicollinearity Reduction: Iterative VIF pruning with a threshold of 8.0, reducing features from 22 to 11.

Feature Normalization: StandardScaler fit strictly on training data and applied to validation/test sets.

Model Training & Tuning: Random Forest Regressor vs. Bagged Decision Tree Regressor (800 estimators each).

Explainable AI: SHAP TreeExplainer, Permutation Importance, and LIME tabular instance explainers.

# Installation & Requirements

Run the following command to install the required dependencies:

pip install numpy pandas scikit-learn matplotlib seaborn statsmodels shap lime openpyxl


# How to Run

Place your dataset file named final_dataset.xlsx in your working directory.

Run the pipeline script:

python main.py


# Cross-Validation & Test Results

Performance Table

Random Forest:

5-Fold CV Mean R²: 0.6318 (+/- 0.1044)

Train R²: 0.9642 | Train RMSE: 0.000789 | Train MAE: 0.000434

Test R²: 0.8620 | Test RMSE: 0.001159 | Test MAE: 0.000712

Bagged Decision Tree:

5-Fold CV Mean R²: 0.6543 (+/- 0.1098)

Train R²: 0.9578 | Train RMSE: 0.000857 | Train MAE: 0.000451

Test R²: 0.8716 | Test RMSE: 0.001118 | Test MAE: 0.000665

Winning Model on Holdout Set: Bagged Decision Tree (R² = 0.8716).

Explainability Insights (XAI)

Multicollinearity: VIF reduced the feature space to 11 key features, keeping UK, AU_Interest_Rate, Gold_Price_Daily_Return, Oil_Price_Daily_Return, lags 1 to 3, and rolling return statistics.

Global Attribution: SHAP summary plots and permutation importance show that short-term return momentum (MA3, Lag1) holds the largest decision weight, with oil and gold returns providing secondary macro signals.

Local Attribution (LIME): For test instance 0, positive momentum in the 3-day moving average contributed strongly toward a positive forecast (+0.00185 for RF, +0.00208 for BDT), while negative values in Lag1 and Lag2 pushed predictions downward.
