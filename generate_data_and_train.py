import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import joblib
import json
import os

np.random.seed(42)

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ── LOAN DATASET ──
n = 2000
age = np.random.randint(22, 65, n)
income = np.random.randint(20000, 150000, n)
credit_score = np.random.randint(300, 850, n)
loan_amount = np.random.randint(5000, 100000, n)
employment_years = np.random.randint(0, 30, n)
debt_to_income = np.round(np.random.uniform(0.05, 0.60, n), 2)
num_credit_lines = np.random.randint(1, 15, n)
missed_payments = np.random.randint(0, 10, n)
education = np.random.choice(['High School','Bachelor','Master','PhD'], n)
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
edu_num = np.array([edu_map[e] for e in education])

score = ((credit_score-300)/550*0.35 + (income/150000)*0.25 +
         (employment_years/30)*0.15 - debt_to_income*0.15 - missed_payments/10*0.10)
prob = 1/(1+np.exp(-10*(score+np.random.normal(0,0.05,n)-0.45)))
approved = (prob>0.5).astype(int)

loan_df = pd.DataFrame({
    'age':age,'income':income,'credit_score':credit_score,'loan_amount':loan_amount,
    'employment_years':employment_years,'debt_to_income':debt_to_income,
    'num_credit_lines':num_credit_lines,'missed_payments':missed_payments,
    'education':education,'education_num':edu_num,'loan_approved':approved
})
loan_df.to_csv('data/sample_loan_data.csv', index=False)

# ── FRAUD DATASET - balanced for better RF performance ──
n_legit = 1800
n_fraud = 200

# Legitimate transactions
legit = pd.DataFrame({
    'txn_amount': np.random.exponential(150, n_legit).clip(1,2000),
    'hour': np.random.randint(8, 22, n_legit),
    'distance_from_home': np.random.exponential(15, n_legit).clip(0,100),
    'avg_daily_spend': np.random.randint(100, 400, n_legit),
    'num_txn_today': np.random.randint(1, 8, n_legit),
    'foreign_txn': np.random.choice([0,1], n_legit, p=[0.97,0.03]),
    'is_fraud': 0
})

# Fraudulent transactions - clearly different patterns
fraud = pd.DataFrame({
    'txn_amount': np.random.exponential(800, n_fraud).clip(200,10000),
    'hour': np.random.choice(list(range(0,6))+list(range(22,24)), n_fraud),
    'distance_from_home': np.random.exponential(200, n_fraud).clip(100,1000),
    'avg_daily_spend': np.random.randint(50, 200, n_fraud),
    'num_txn_today': np.random.randint(10, 30, n_fraud),
    'foreign_txn': np.random.choice([0,1], n_fraud, p=[0.4,0.6]),
    'is_fraud': 1
})

fraud_df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
fraud_df['txn_amount'] = fraud_df['txn_amount'].round(2)
fraud_df['distance_from_home'] = fraud_df['distance_from_home'].round(1)
fraud_df.to_csv('data/sample_fraud_data.csv', index=False)

print(f"Loan data: {len(loan_df)} rows, {loan_df['loan_approved'].sum()} approved")
print(f"Fraud data: {len(fraud_df)} rows, {fraud_df['is_fraud'].sum()} fraud cases")

# ── TRAIN LOAN MODEL ──
loan_features = ['age','income','credit_score','loan_amount','employment_years',
                 'debt_to_income','num_credit_lines','missed_payments','education_num']
X_loan = loan_df[loan_features]
y_loan = loan_df['loan_approved']
X_tr,X_te,y_tr,y_te = train_test_split(X_loan,y_loan,test_size=0.2,random_state=42)
scaler_loan = StandardScaler()
X_tr_s = scaler_loan.fit_transform(X_tr)
X_te_s = scaler_loan.transform(X_te)
rf_loan = RandomForestClassifier(n_estimators=100,max_depth=8,random_state=42)
rf_loan.fit(X_tr_s,y_tr)
lp = rf_loan.predict_proba(X_te_s)[:,1]
loan_auc = roc_auc_score(y_te,lp)
loan_pred = rf_loan.predict(X_te_s)
loan_cm = confusion_matrix(y_te,loan_pred).tolist()
loan_rep = classification_report(y_te,loan_pred,output_dict=True)
loan_imp = dict(zip(loan_features,rf_loan.feature_importances_.tolist()))
print(f"Loan AUC: {loan_auc:.4f}")

# ── TRAIN FRAUD MODEL ──
fraud_features = ['txn_amount','hour','distance_from_home','avg_daily_spend','num_txn_today','foreign_txn']
X_fraud = fraud_df[fraud_features]
y_fraud = fraud_df['is_fraud']
X_tr2,X_te2,y_tr2,y_te2 = train_test_split(X_fraud,y_fraud,test_size=0.2,random_state=42)
scaler_fraud = StandardScaler()
X_tr2_s = scaler_fraud.fit_transform(X_tr2)
X_te2_s = scaler_fraud.transform(X_te2)
rf_fraud = RandomForestClassifier(n_estimators=150,max_depth=8,random_state=42,class_weight='balanced')
rf_fraud.fit(X_tr2_s,y_tr2)
fp = rf_fraud.predict_proba(X_te2_s)[:,1]
fraud_auc = roc_auc_score(y_te2,fp)
fraud_pred = rf_fraud.predict(X_te2_s)
fraud_cm = confusion_matrix(y_te2,fraud_pred).tolist()
fraud_rep = classification_report(y_te2,fraud_pred,output_dict=True)
fraud_imp = dict(zip(fraud_features,rf_fraud.feature_importances_.tolist()))
print(f"Fraud AUC: {fraud_auc:.4f}")

# ── SAVE ──
joblib.dump(rf_loan,'models/loan_model.pkl')
joblib.dump(scaler_loan,'models/loan_scaler.pkl')
joblib.dump(rf_fraud,'models/fraud_model.pkl')
joblib.dump(scaler_fraud,'models/fraud_scaler.pkl')

fpr,tpr,_ = roc_curve(y_te,lp)
loan_roc=[{"fpr":round(float(f),3),"tpr":round(float(t),3)} for f,t in zip(fpr[::5],tpr[::5])]
fpr2,tpr2,_ = roc_curve(y_te2,fp)
fraud_roc=[{"fpr":round(float(f),3),"tpr":round(float(t),3)} for f,t in zip(fpr2[::5],tpr2[::5])]

def get_metric(rep, key):
    for k in [str(key), key, '1', 1]:
        if k in rep and isinstance(rep[k], dict):
            return rep[k]
    return {}

metrics = {
    "loan": {
        "auc": round(loan_auc,4), "accuracy": round(loan_rep['accuracy'],4),
        "precision": round(get_metric(loan_rep,1).get('precision',0),4),
        "recall": round(get_metric(loan_rep,1).get('recall',0),4),
        "f1": round(get_metric(loan_rep,1).get('f1-score',0),4),
        "confusion_matrix": loan_cm, "feature_importances": loan_imp, "roc_points": loan_roc
    },
    "fraud": {
        "auc": round(fraud_auc,4), "accuracy": round(fraud_rep['accuracy'],4),
        "precision": round(get_metric(fraud_rep,1).get('precision',0),4),
        "recall": round(get_metric(fraud_rep,1).get('recall',0),4),
        "f1": round(get_metric(fraud_rep,1).get('f1-score',0),4),
        "confusion_matrix": fraud_cm, "feature_importances": fraud_imp, "roc_points": fraud_roc
    },
    "dataset": {
        "total_records": len(loan_df),
        "loan_approved_pct": round(loan_df['loan_approved'].mean()*100,1),
        "fraud_pct": round(fraud_df['is_fraud'].mean()*100,1)
    }
}

with open('models/metrics.json','w') as f:
    json.dump(metrics,f,indent=2)

print("\nAll models and metrics saved!")
