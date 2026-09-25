# telecom-customer-churn-prediction
End-to-end customer churn prediction using Python and Scikit-learn, covering data preprocessing, EDA, feature engineering, model training, evaluation, and prediction.
Now I have the full picture of the notebook. Here's a README section for your GitHub project:

---

# Telecom Customer Churn Prediction

## 📌 Overview
This project analyzes a telecom company's customer dataset to understand the key drivers of customer churn and builds a machine learning model to predict whether a customer is likely to churn. The goal is to help the business proactively identify at-risk customers and take retention action.

## 📊 Dataset
The dataset (`Telecom_customer_churn.csv`) contains customer-level information including:
- **Demographics:** gender, senior citizen status, partner, dependents
- **Account info:** tenure, contract type, payment method, billing preferences
- **Services:** phone, internet, streaming, security, tech support, etc.
- **Charges:** monthly charges, total charges
- **Target:** `Churn` (Yes/No)

## 🛠️ Project Workflow

### 1. Data Cleaning
- Dropped the non-informative `customerID` column
- Removed 22 duplicate records
- Fixed `TotalCharges` (converted from object to float, handled whitespace values)
- Imputed missing values in `TotalCharges` using the mean

### 2. Exploratory Data Analysis (EDA)
- Analyzed distribution and skewness of numerical features
- Visualized churn patterns across gender, senior citizen status, partner, and dependents
- Explored churn distribution using pie charts and count plots

### 3. Data Preprocessing
- Label-encoded all categorical variables
- Detected and removed outliers using the **Z-score method**
- Dropped the `PhoneService` column (low predictive value)
- Corrected skewness in `TotalCharges` using a **Yeo-Johnson Power Transform**
- Balanced the imbalanced target class using **SMOTE (oversampling)**
- Scaled features using **StandardScaler**

### 4. Model Building & Evaluation
Trained and compared three classification models, selecting the best random state for each:

| Model | Accuracy |
|---|---|
| Logistic Regression | 78% |
| **Random Forest Classifier** | **84%** |
| Decision Tree Classifier | 79% |

### 5. Hyperparameter Tuning
Used **GridSearchCV** to tune the Decision Tree Classifier:
```
Best params: {'criterion': 'gini', 'max_depth': 9, 'min_samples_leaf': 6, 'min_samples_split': 19}
```

### 6. Final Model
A tuned Decision Tree Classifier was used as the final model, achieving an accuracy of **~78%** on the test set.

## 🧰 Tech Stack
- **Python**
- **Pandas, NumPy** – data manipulation
- **Matplotlib, Seaborn** – visualization
- **Scikit-learn** – preprocessing, modeling, evaluation
- **Imbalanced-learn (SMOTE)** – handling class imbalance

## 🚀 How to Run
1. Clone the repository
2. Install dependencies: `pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn`
3. Place `Telecom_customer_churn.csv` in the project directory
4. Run the notebook `Telecom_customer_churn_prediction_project.ipynb`

## 📈 Key Insights
- Random Forest gave the best overall accuracy among the tested models
- Contract type, tenure, and monthly charges were among the most influential churn indicators
- Class imbalance in the target variable was effectively addressed using SMOTE

---
