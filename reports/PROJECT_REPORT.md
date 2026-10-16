# Real-Time Fraud Detection Using Predictive Models

## 1. Introduction

This project develops a machine learning system to classify financial transactions as potentially fraudulent or legitimate. It uses the BankSim dataset to train classification models and exposes the selected model through a FastAPI application.

The project demonstrates the workflow from data preparation and model evaluation to saving a trained pipeline and serving predictions through an API.

## 2. Problem Statement

Fraudulent transactions represent a small proportion of the transactions in the dataset. A useful fraud detection model must identify fraudulent transactions while limiting false alarms on legitimate transactions.

The project explores classification algorithms and evaluates their performance using precision, recall, F1 score, accuracy, and confusion matrix values.

## 3. Dataset and Preparation

The project uses the BankSim synthetic transaction dataset.

The original dataset contains transaction time step, customer identifier, age category, gender, merchant, transaction category, amount, and fraud label, alongside two constant location-code columns.

Data preparation included:
- Inspecting the dataset structure and data quality.
- Checking missing values and duplicate records.
- Removing constant columns.
- Saving the processed dataset separately from the raw data.
- Splitting the data into training and test datasets using stratification.

The training dataset contains 475,714 transactions, and the held-out test dataset contains 118,929 transactions.

## 4. Model Development

Two classification models were developed:

1. Logistic Regression
2. Random Forest

The preprocessing pipeline uses one-hot encoding for categorical features and standard scaling for numerical features. The raw customer identifier was excluded from model inputs.

The models were compared using validation data and different probability thresholds. Random Forest was selected for final evaluation based on the experiments conducted.

## 5. Final Random Forest Evaluation

The selected Random Forest pipeline was trained on the complete training dataset and evaluated on the separate test dataset using a decision threshold of 0.60.

| Metric | Result |
|---|---:|
| Accuracy | 99.48% |
| Fraud precision | 77.89% |
| Fraud recall | 80.00% |
| Fraud F1 score | 0.7893 |
| True positives | 1,152 |
| False positives | 327 |
| True negatives | 117,162 |
| False negatives | 288 |

### Interpretation

- **Precision of 77.89%:** Among transactions classified as fraud, approximately 77.89% were actually fraudulent in the test dataset.
- **Recall of 80.00%:** The model detected 80% of the fraudulent transactions in the test dataset.
- **False positives:** 327 legitimate transactions were classified as fraud.
- **False negatives:** 288 fraudulent transactions were classified as legitimate.

Accuracy should not be interpreted alone because fraudulent transactions form a small minority of the dataset.

## 6. Model Persistence

The fitted pipeline and decision threshold were saved for reuse.

- `models/random_forest_pipeline.joblib` stores the fitted preprocessing pipeline and Random Forest classifier.
- `models/decision_threshold.txt` stores the selected threshold of 0.60.

The saved pipeline can be loaded without retraining it. A separate script verified that the saved model could generate predictions.

## 7. FastAPI Integration

A FastAPI application was created in `src/api/main.py`.

The application:
1. Loads the saved model pipeline and threshold.
2. Defines a transaction request schema.
3. Provides a health-check endpoint.
4. Accepts transaction details through `POST /predict`.
5. Calculates the fraud probability using the saved pipeline.
6. Applies the threshold to classify the transaction.
7. Returns a JSON response.

The prediction response includes:
- Fraud probability.
- Decision threshold.
- Numeric prediction.
- Human-readable result.

## 8. API Testing

Testing through the interactive FastAPI documentation confirmed the following:

- A valid transaction returned HTTP 200 and a prediction.
- A negative transaction amount was rejected with HTTP 422.
- A request missing the required merchant field was rejected with HTTP 422.

These tests demonstrate that the prediction endpoint works for the tested input and that basic request validation is functioning.

## 9. Project Limitations

- BankSim is a synthetic dataset and may not fully represent real banking behavior.
- The API runs locally and is not connected to a live banking transaction stream.
- Predictions are estimates, not guarantees that a transaction is fraudulent or legitimate.
- The selected threshold reflects the experiments performed and may not be suitable for every business scenario.
- Production deployment would require additional security, monitoring, operational testing, and validation on representative real-world data.

## 10. Future Enhancements

Possible improvements include:
- More extensive automated tests.
- Stronger validation of categorical inputs.
- A frontend for entering transactions and viewing predictions.
- Deployment to a suitable hosting platform.
- Monitoring for changes in transaction patterns and model performance.
- Evaluation on additional representative data.

## 11. Conclusion

The project demonstrates an end-to-end machine learning workflow for fraud detection, including data preparation, classification model comparison, final test evaluation, model persistence, and API-based prediction.

The resulting application provides a foundation for further development and experimentation with machine learning-based fraud detection.
