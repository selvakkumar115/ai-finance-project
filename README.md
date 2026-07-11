# 🏦 AI-Powered Financial Decision System

**Final Year Project | B.Sc Artificial Intelligence & Data Science**  
KPR College of Arts, Science and Research

---

## 📋 Project Overview

An end-to-end AI system for financial decision-making featuring:
- **Loan Approval Prediction** — Random Forest (AUC: 0.93+)
- **Fraud Detection** — Gradient Boosting (AUC: 0.99+)
- **Streamlit Web App** — Login, CSV upload, charts, downloads

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models (generates .pkl files)
```bash
python generate_data_and_train.py
```

### 3. Launch the web application
```bash
streamlit run app.py
```

### 4. Login credentials
| Username | Password   | Role    |
|----------|------------|---------|
| admin    | admin123   | Admin   |
| analyst  | analyst123 | Analyst |
| demo     | demo       | Viewer  |

---

## 📁 Project Structure

```
ai_finance_project/
│
├── app.py                          # Main Streamlit application
├── generate_data_and_train.py      # Data generation + model training
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── models/
│   ├── loan_model.pkl              # Trained Random Forest model
│   ├── loan_scaler.pkl             # StandardScaler for loan features
│   ├── fraud_model.pkl             # Trained Gradient Boosting model
│   ├── fraud_scaler.pkl            # StandardScaler for fraud features
│   └── metrics.json                # Model evaluation metrics
│
├── data/
│   ├── sample_loan_data.csv        # Sample loan dataset (2000 rows)
│   └── sample_fraud_data.csv       # Sample fraud dataset (2000 rows)
│
├── AI_Financial_Decision_System_Report.docx       # Project report
└── AI_Financial_Decision_System_Presentation.pptx # Slides (13 slides)
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 Login & Registration | Role-based access: Admin / Analyst / Viewer |
| 📂 CSV Upload | Bulk upload with automatic cleaning & prediction |
| 🧹 Auto Data Cleaning | Null imputation, type casting, range clipping |
| 🤖 Loan Prediction | Random Forest with 9 features, instant result |
| 💳 Fraud Detection | Gradient Boosting, real-time risk scoring |
| 📊 Dashboard | KPI cards, model accuracy charts |
| 📈 ROC & Confusion Matrix | Interactive performance charts |
| 📥 Download Results | Export predictions as CSV |
| 👤 Individual Analysis | Full risk profile per customer |
| 💡 AI Recommendations | Actionable advice per prediction |
| 💾 Saved Models | .pkl files for instant deployment |

---

## 🤖 Models

### Loan Approval — Random Forest
- **Features:** age, income, credit_score, loan_amount, employment_years, debt_to_income, num_credit_lines, missed_payments, education
- **Algorithm:** RandomForestClassifier (100 trees, max_depth=8)
- **AUC:** 0.932+

### Fraud Detection — Gradient Boosting
- **Features:** txn_amount, hour, distance_from_home, avg_daily_spend, num_txn_today, foreign_txn
- **Algorithm:** GradientBoostingClassifier (100 trees, max_depth=5)
- **AUC:** 0.997+

---

## 📚 References

- Altman (1968) — Credit scoring discriminant analysis
- Chen & Guestrin (2016) — XGBoost gradient boosting  
- Fischer & Krauss (2018) — LSTM for financial prediction
- Lundberg & Lee (2017) — SHAP explainability values
- Soniya T et al. (2024) — AI/ML in Financial Decision Systems (Chapter)

---

## 👥 Authors

- Soniya T — soniya.25bscaids@kprcas.ac.in
- Sathurthika Devi S — sathurthikadevi.25bscaids@kprcas.ac.in
- Selvakkumar TK — selvakkumar.25bscaids@kprcas.ac.in
- Sri Anbukarasi D — srianbukarasi.25bscaids@kprcas.ac.in
