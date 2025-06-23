# 💳 Fraud Detection System using Machine Learning & Streamlit

This project is an end-to-end implementation of a fraud detection system using machine learning. It includes data preprocessing, model training, performance evaluation, and deployment via a user-friendly Streamlit web application.

---

## 🚀 Project Overview

The goal of this project is to build a system that detects fraudulent transactions from financial data. Since online fraud is increasing rapidly, an automated, intelligent system that can flag suspicious transactions is critical.

---

## 📌 Features

- Preprocessed and analyzed a real-world transactions dataset.
- Implemented multiple ML models: Logistic Regression, Decision Tree, Random Forest, XGBoost.
- Evaluated models using accuracy, recall, precision, F1-score, and ROC AUC.
- Selected and saved the best-performing model.
- Built a clean UI using Streamlit for real-time fraud predictions.
- Model deployed locally (and can be extended to cloud).

---

## 🧠 Machine Learning Models Used

- Logistic Regression (with class imbalance handling)
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Model performance was compared, and the best one was used in the final app based on:
- Recall for fraud class
- ROC AUC Score

---

## 📊 Input Features

- `type`: Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT)
- `amount`: Transaction amount
- `oldbalanceOrg`: Original sender balance
- `newbalanceOrig`: New sender balance
- `oldbalanceDest`: Original receiver balance
- `newbalanceDest`: New receiver balance

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy, Scikit-learn
- XGBoost
- Streamlit (for deployment)
- Joblib (for saving model)

---

## 📁 Directory Structure

