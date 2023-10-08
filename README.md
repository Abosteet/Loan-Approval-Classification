# Loan Approval Classification 🏦

> A machine learning project that classifies loan approval status using the Loan Prediction dataset, comparing nine classification models including K-Fold cross-validation and ensemble techniques (Voting, Bagging, Stacking).

---

## 📌 Overview

Predicting loan approval is a critical task for financial institutions, enabling faster and more consistent credit decisions. This project builds and evaluates a comprehensive set of machine learning models to classify whether a loan application will be approved or not, based on applicant information including income, credit history, loan amount, and demographics.

The project goes beyond basic model training by implementing a data-driven approach to hyperparameter selection using error rate plots, alongside advanced ensemble techniques including Voting, Bagging, and Stacking classifiers, making it a comprehensive benchmark for loan approval classification.

---

## 🔍 Key Results

| Model | Validation Method | Notes |
|-------|------------------|-------|
| SVC | 9-Fold Cross Validation | Optimal folds selected data-driven |
| KNN | 9-Fold Cross Validation | Best n_neighbors selected via error rate plot |
| Decision Tree | 9-Fold Cross Validation | Best max_depth selected via error rate plot |
| Logistic Regression | 9-Fold Cross Validation | Default parameters |
| Random Forest | 9-Fold Cross Validation | Best n_estimators selected via error rate plot |
| Voting Classifier | Train/Test Split (80/20) | Combines SVM, Decision Tree, LR, KNN |
| Bagging Classifier | Train/Test Split (80/20) | Best base estimator and n_estimators selected data-driven |
| Stacking Classifier | Train/Test Split (80/20) | Best final estimator selected data-driven |
| Naive Bayes | Train/Test Split (80/20) | GaussianNB |

> Stacking Classifier achieved the best accuracy at 90%.

---

## ⚙️ Project Pipeline

### 1. Data Loading and Cleaning
- Loaded the Loan Prediction dataset
- Dropped the `Loan_ID` column
- Inspected shape, data types, unique values, and class distribution
- Filled missing values: mean for `LoanAmount`, mode for `Gender`, `Married`, `Loan_Amount_Term`, `Credit_History`, `Self_Employed`, and `Dependents`
- Verified zero duplicate rows
- Applied Label Encoding for all categorical columns
- Visualized outliers using box plots before and after handling
- Applied IQR-based outlier capping for all columns
- Dropped columns with a single unique value after outlier handling

### 2. Exploratory Data Analysis
- Distribution histograms for all features
- Pairplot split by Loan_Status
- Scatter plots of ApplicantIncome vs CoapplicantIncome and ApplicantIncome vs LoanAmount split by Loan_Status
- Pie chart for Loan_Status distribution using plotly
- Automated EDA report using pandas-profiling

### 3. Data Preprocessing
- Oversampled minority class (Loan_Status = 0) using `resample` to balance the dataset
- Separated features and target column
- Applied StandardScaler for feature scaling

### 4. Optimal Fold Selection
- Tested K-Fold cross-validation from 2 to 10 folds using Random Forest
- Plotted overall accuracy per number of folds to determine the optimal split

### 5. Model Training and Evaluation
- Trained SVC, KNN, Decision Tree, Logistic Regression, and Random Forest using 9-Fold cross-validation
- Tuned hyperparameters for KNN (best n_neighbors), Decision Tree (best max_depth), and Random Forest (best n_estimators) using error rate plots
- Decision Tree visualization using sklearn tree plotter
- Implemented Voting, Bagging, and Stacking ensemble classifiers with data-driven estimator selection
- Trained Naive Bayes with confusion matrix and classification report

---

## 🛠️ Tools and Libraries

| Tool | Usage |
|------|-------|
| **pandas** | Data loading, cleaning, and manipulation |
| **numpy** | Numerical operations |
| **matplotlib / seaborn / plotly** | Data visualization |
| **scikit-learn** | Models, cross-validation, ensemble methods, scaling, and metrics |
| **pandas-profiling** | Automated EDA report generation |

---

## 📂 Dataset

This project uses the **Loan Prediction** dataset from Kaggle.

🔗 [View Dataset on Kaggle](https://www.kaggle.com/datasets/leonbora/analytics-vidhya-loan-prediction)
