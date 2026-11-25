# Customer Churn Prediction — Telco Dataset

Binary classification project predicting customer churn on the IBM Telco Customer Churn dataset (7,043 customers, ~26% churn rate).

## What this covers

- Data cleaning (`TotalCharges` type coercion, missing-value handling)
- EDA on churn rate and class imbalance
- Feature engineering: numeric scaling + one-hot encoding via `ColumnTransformer`
- Class-imbalance handling with **SMOTE** (applied to the training set only)
- Three models compared: Logistic Regression, Random Forest, LightGBM
- Evaluation with **precision, recall, F1, and ROC-AUC** — not just accuracy, since the dataset is imbalanced (a model that always predicts "no churn" would already look ~73% accurate while being useless)
- Feature importance to answer *why* customers churn, not just whether they will

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.50 | 0.78 | 0.61 | 0.83 |
| Random Forest | 0.58 | 0.58 | 0.58 | 0.82 |
| LightGBM | 0.60 | 0.60 | 0.60 | 0.83 |

*(Exact numbers can vary slightly by run/random seed.)*

## Key churn drivers

Contract type (month-to-month vs. long-term), tenure, monthly charges, and internet/tech-support service flags were the strongest predictors on this dataset.

## How to run

Open `churn_prediction.ipynb` in Google Colab (or Jupyter) and run all cells top to bottom. The dataset is pulled directly from a public URL — no download or Kaggle API key needed.

```
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb
```

## Tech stack

Python, pandas, scikit-learn, imbalanced-learn (SMOTE), LightGBM, matplotlib/seaborn
