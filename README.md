# Customer Churn Prediction (Telecom)

Predict which telecom customers are likely to leave, explain why, and turn the predictions into a retention strategy.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 20 features, 26.5% churn)
**Models:** Logistic Regression, Random Forest, XGBoost (tuned with randomized search)
**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

## Project structure
```
churn_prediction.ipynb   main analysis (EDA, modelling, evaluation, SHAP)
app.py                   Streamlit app for single-customer predictions
data/                    Telco-Customer-Churn.csv
models/                  saved model + results table
images/                  exported charts
requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
jupyter notebook churn_prediction.ipynb     # run all cells
streamlit run app.py                        # needs models/churn_model.joblib from the notebook
```

## Key findings (EDA)
| Segment | Churn rate |
|---|---|
| Month-to-month contract | 42.7% |
| One-year contract | 11.3% |
| Two-year contract | 2.8% |
| Fiber optic internet | 41.9% (DSL 19.0%) |
| No Tech Support (with internet) | 41.6% (with it: 15.2%) |
| Electronic check payment | 45.3% (automatic payments: 15-17%) |
| Tenure 0-6 months | 52.9% (49+ months: 9.5%) |

The highest-risk group is **month-to-month, fiber optic customers in their first year: 916 customers with a 70.2% churn rate.**

## Model results (hold-out test set, threshold 0.5)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.737 | 0.503 | 0.783 | 0.613 | 0.842 |
| Random Forest | 0.759 | 0.531 | 0.775 | 0.630 | 0.841 |
| XGBoost | 0.750 | 0.519 | 0.794 | 0.628 | 0.843 |
| **XGBoost (tuned)** | 0.745 | 0.513 | **0.799** | 0.625 | **0.848** |

A model that always predicts "no churn" scores 73.5% accuracy, so accuracy is not the headline metric here. Recall on churners and ROC-AUC are. 5-fold CV ROC-AUC for the tuned XGBoost is 0.850 ± 0.011.

## Threshold tuning
With assumed costs of 500 per missed churner and 50 per unnecessary retention offer, the cost-optimal threshold is **0.17**. It catches 97% of churners (precision falls to 38%) and cuts the total cost from 51,700 to 34,950 on the test set. The cost figures are assumptions; replace them with real customer lifetime value and offer costs.

## Recommendations
1. Incentivise month-to-month customers to move to longer contracts, especially in the first 6 months.
2. Build a 90-day onboarding programme, since early tenure is the danger zone.
3. Investigate fiber optic pricing and service quality.
4. Bundle Tech Support and Online Security for high-risk customers.
5. Encourage a switch from electronic check to automatic payments.
6. Score customers weekly and target the highest-risk ones with retention offers.

## Limitations
- Single snapshot dataset with no time dimension, so it cannot capture seasonality or drift.
- The cost threshold was picked on the test set for demonstration; use a validation split in production.
- Retention offers should be A/B tested to confirm they actually reduce churn.
