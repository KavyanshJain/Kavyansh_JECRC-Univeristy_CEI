# Customer Intelligence System — Bank Churn Analysis

End-to-end ML pipeline on the Bank Customer Churn dataset covering classification, ensemble learning, and customer segmentation.

Built as part of the **Celebal Technologies Data Science Training** (Week 3).

---

## What's in here

- Exploratory data analysis on 10,000 bank customers
- Baseline classification with Logistic Regression
- Ensemble models — Random Forest and XGBoost with GridSearchCV tuning
- Customer segmentation using K-Means and DBSCAN
- Feature importance analysis and model comparison

## Dataset

[Bank Customer Churn Prediction](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) from Kaggle (`shrutimechlearn/churn-modelling`). 10,000 rows, 14 columns, ~20% churn rate.

## Results

| Model | ROC-AUC | F1 (churn class) |
|---|---|---|
| Logistic Regression | 0.769 | 0.25 |
| Random Forest | 0.857 | 0.61 |
| XGBoost | 0.866 | 0.60 |

XGBoost came out on top. Key predictors were `Age`, `NumOfProducts`, and `IsActiveMember`.

## Setup

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost kagglehub
```

The notebook loads the dataset directly via `kagglehub` — no manual download needed.

## Stack

Python · scikit-learn · XGBoost · pandas · matplotlib · seaborn
