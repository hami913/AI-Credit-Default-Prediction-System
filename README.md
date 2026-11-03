# AI Credit Card Default Prediction System

An end-to-end Machine Learning project that predicts the probability of credit card payment default using customer demographic and historical payment information.

The system uses a Random Forest classifier, feature engineering, model evaluation, and an interactive Streamlit application to support credit risk assessment.

## Project Overview

Credit card default is a major challenge for financial institutions. Identifying customers who may fail to make their payments helps financial organizations assess credit risk and make better-informed decisions.

This project develops a Machine Learning-based credit default prediction system using the UCI Credit Card Default dataset.

### Key Features

- Credit card default prediction
- Data preprocessing and exploratory data analysis
- Feature engineering
- Random Forest classification
- Probability-based risk assessment
- Interactive Streamlit web application
- Model evaluation and performance analysis
- Automated testing

## Dataset Information

**Dataset:** UCI Credit Card Default Dataset

**Dataset Size:** 30,000 records

**Original Features:** 25 columns

**Target Variable:** `default.payment.next.month`

The dataset contains customer demographic information, credit limits, payment history, bill statements, and previous payment amounts.

### Target Distribution

| Class | Description | Records |
|---|---|---:|
| 0 | No Default | 23,364 |
| 1 | Default | 6,636 |
| Total | All Customers | 30,000 |

- No Default: 77.88%
- Default: 22.12%

The dataset contains an imbalanced target distribution, making appropriate evaluation metrics important.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Pytest
- Git and GitHub

## Machine Learning Workflow

The project follows a complete Machine Learning pipeline:

1. Data Collection
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Cleaning and Preprocessing
5. Feature Engineering
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Saving
10. Streamlit Deployment

## Feature Engineering

The original dataset contains 23 predictive input features after excluding the customer ID and target variable.

Feature engineering was applied to create additional meaningful predictive variables.

- Original predictive features: 23
- Engineered features: 38

The engineered features help the model capture additional patterns in customer credit behavior.

## Model Development

**Algorithm:** Random Forest Classifier

The Random Forest algorithm was selected for its ability to model nonlinear relationships and work with structured tabular data.

### Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Random Forest |
| Input Features | 23 |
| Engineered Features | 38 |
| Training Data | 24,000 |
| Testing Data | 6,000 |
| Train-Test Split | 80:20 |
| Stratification | Enabled |
| Classification Threshold | 0.50 |

The dataset was divided using stratified sampling to maintain the target class distribution in training and testing sets.

## Model Performance

The final Random Forest model achieved the following test-set results:

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 77.68% |
| Precision | 49.62% |
| Recall | 58.63% |
| F1-Score | 53.75% |
| ROC-AUC | 77.77% |
| PR-AUC | 55.48% |

### Cross-Validation Results

Out-of-fold evaluation:

| Metric | Score |
|---|---:|
| OOF F1-Score | 54.72% |
| OOF ROC-AUC | 78.55% |

These metrics provide a more comprehensive understanding of model performance, particularly because the dataset is imbalanced.

## Streamlit Web Application

The project includes an interactive Streamlit application that allows users to enter customer information and obtain a credit default risk prediction.

### Application Features

- Customer information input
- Credit default prediction
- Probability-based risk assessment
- High-risk and lower-risk classification
- Interactive user interface
- Model-based prediction results

The application uses the trained Random Forest model saved in the project.

## Saved Model

Model file:

`models/credit_default_model_v1.joblib`

Model information:

- Model: Credit Default Random Forest
- Version: 1.0
- Classification threshold: 0.50

Joblib is used to save and load the trained Machine Learning model.

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hami913/AI-Credit-Default-Prediction-System.git
```

### 2. Navigate to the Project

```bash
cd AI-Credit-Default-Prediction-System
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run app.py
```

Note: Run the command using the actual Streamlit entry-point filename in your repository if it differs from `app.py`.

## Model Testing

Automated testing was implemented using Pytest to verify the project functionality.

**Latest test results:**

- 23 test cases
- 23 passed
- 0 failed

Run tests using:

```bash
pytest
```

## Model Deployment

The project is deployed using Streamlit Cloud.

The application is connected to the GitHub repository, allowing updates to be deployed through repository changes.

## Project Repository

GitHub:

https://github.com/hami913/AI-Credit-Default-Prediction-System

## Future Improvements

- Hyperparameter optimization
- Advanced ensemble learning techniques
- Explainable AI using SHAP
- Improved recall for default customers
- Model monitoring and performance tracking
- Advanced credit risk visualization
- Further model calibration and threshold optimization

## Disclaimer

This project is developed for educational and Machine Learning research purposes. Its predictions should not be considered the sole basis for real-world financial or lending decisions.

## Author

**Mian Hamza Ahmad**

BS Artificial Intelligence Student

University of Lahore

GitHub: https://github.com/hami913

LinkedIn: https://linkedin.com/in/mian-hamza-ahmad-8536773a3

---

If you find this project useful, consider giving the repository a star!
