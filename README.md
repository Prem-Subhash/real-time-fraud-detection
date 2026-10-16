# Real-Time Fraud Detection Using Predictive Models

## 1. Project Overview

This project uses machine learning to identify potentially fraudulent financial transactions.

The project uses the BankSim dataset to train and evaluate classification models. Logistic Regression and Random Forest were compared, and the selected Random Forest pipeline was saved and integrated with a FastAPI application.

The API accepts transaction details and returns a predicted fraud probability and classification.

**Note:** This project currently provides predictions through an API. It is not connected to a live banking transaction stream.

## 2. Project Objectives

- Prepare and explore transaction data.
- Train Logistic Regression and Random Forest classifiers.
- Compare models using appropriate classification metrics.
- Evaluate the selected model on a separate test dataset.
- Save the trained model and decision threshold.
- Expose predictions through a FastAPI endpoint.
- Test valid and invalid API requests.

## 3. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Joblib
- Pydantic
- VS Code

## 4. Dataset

The project uses the BankSim synthetic banking transaction dataset.

The original dataset contains transaction information such as:

- `step`
- `customer`
- `age`
- `gender`
- `zipcodeOri`
- `merchant`
- `zipMerchant`
- `category`
- `amount`
- `fraud`

The target column is `fraud`:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

During data preparation, missing values and duplicate records were checked, and constant columns were removed. The processed dataset contains eight columns, including the target.

The raw dataset is kept separate from the processed data.

## 5. Machine Learning Workflow

### Data Preparation

The raw dataset was inspected and prepared for model development. Constant columns were removed, and the processed data was saved.

### Train-Test Split

The processed dataset was divided into:

- Training dataset: 475,714 transactions
- Test dataset: 118,929 transactions

Stratified splitting was used to preserve the fraud-to-legitimate class distribution.

### Preprocessing

The model pipeline uses:

- One-hot encoding for categorical features.
- Standard scaling for numerical features.
- A Random Forest classifier for prediction.

The raw customer identifier was excluded from the model inputs.

### Model Training and Comparison

Two classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest

Validation results and different decision thresholds were compared. Random Forest was selected for the final evaluation based on the experiments conducted.

### Final Evaluation

The selected Random Forest pipeline was trained on the complete training dataset and evaluated on the separate test dataset.

## 6. Final Random Forest Results

The final test evaluation used a decision threshold of `0.60`.

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

These results are from the project's held-out test dataset. The high accuracy should be interpreted alongside precision and recall because fraudulent transactions are a small minority of the dataset.

## 7. Saved Model Files

The trained artifacts are stored in the `models/` directory.

- `random_forest_pipeline.joblib`: Contains the fitted preprocessing pipeline and Random Forest classifier.
- `decision_threshold.txt`: Contains the selected decision threshold, `0.60`.

The API loads these artifacts instead of retraining the model for each prediction request.

## 8. Project Structure

```text
real-time-fraud-detection-manual/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── random_forest_pipeline.joblib
│   └── decision_threshold.txt
├── notebooks/
├── reports/
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data_processing/
│   ├── training/
│   ├── check_data_quality.py
│   ├── inspect_data.py
│   └── investigate_*.py
├── tests/
│   ├── data_tests/
│   └── model_tests/
├── requirements.txt
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python installed on your system.
- A terminal or command prompt.
- The project files and saved model artifacts.

### Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Ensure that the saved model files exist in the `models/` directory before starting the API.

## 10. Running the API

From the project root, run:

```powershell
python -m uvicorn src.api.main:app --reload
```

Once the server starts, open:

- API home: http://127.0.0.1:8000/
- Interactive API documentation: http://127.0.0.1:8000/docs

Keep the terminal running while testing the API.

## 11. Making a Prediction

Open the interactive API documentation and expand `POST /predict`.

Select **Try it out**, enter a transaction request, and click **Execute**.

Example request:

```json
{
  "step": 25,
  "age": "3",
  "gender": "M",
  "merchant": "M1823072687",
  "category": "es_transportation",
  "amount": 150.0
}
```

The API returns a JSON response containing:

- `fraud_probability`: Estimated probability of fraud.
- `threshold`: Decision threshold used for classification.
- `prediction`: `1` for fraud or `0` for legitimate.
- `result`: Human-readable classification.

The probability and classification depend on the trained model's output.

## 12. API Validation Tests

The following checks were performed through the interactive API documentation:

- A valid transaction returned a successful prediction response.
- A negative transaction amount was rejected with HTTP 422.
- A request missing the required `merchant` field was rejected with HTTP 422.

These tests demonstrate basic request validation and successful prediction execution.

## 13. Limitations

- BankSim is a synthetic dataset and may not represent real-world banking transactions.
- The API currently runs locally and is not connected to a live transaction source.
- Model predictions are estimates, not guarantees that a transaction is fraudulent or legitimate.
- The selected threshold reflects the experiments conducted in this project and may not be appropriate for every business.
- Production deployment would require further validation, security controls, monitoring, and integration with an authorized transaction source.

## 14. Future Improvements

- Add automated unit and integration tests.
- Improve validation of categorical inputs.
- Build a frontend for entering transaction details and displaying predictions.
- Deploy the API to a suitable hosting platform.
- Monitor prediction performance and investigate model drift.
- Evaluate the model on additional representative transaction data.

## 15. Conclusion

This project demonstrates an end-to-end machine learning workflow, from transaction data preparation and model evaluation to saving a trained pipeline and exposing predictions through a FastAPI service.

It provides a foundation for further experimentation with fraud detection and API-based machine learning applications.
