"""
AI-Powered Financial Decision System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
import io
import base64
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #38bdf8; }
    .metric-label { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #0ea5e9, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 15px;
    }

    /* Approval badge */
    .badge-approved {
        background: #065f46; color: #6ee7b7;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
    }
    .badge-rejected {
        background: #7f1d1d; color: #fca5a5;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
    }
    .badge-fraud {
        background: #7f1d1d; color: #fca5a5;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
    }
    .badge-legit {
        background: #065f46; color: #6ee7b7;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
    }

    /* Info box */
    .info-box {
        background: #1e293b; border-left: 4px solid #0ea5e9;
        padding: 12px 16px; border-radius: 0 8px 8px 0; margin: 10px 0;
        color: #e2e8f0;
    }

    /* Risk bar */
    .risk-bar-container { background: #1e293b; border-radius: 8px; height: 12px; margin: 8px 0; }

    /* Login box */
    .login-container {
        max-width: 420px; margin: 80px auto;
        background: #1e293b; padding: 40px;
        border-radius: 16px; border: 1px solid #334155;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0f172a; border-right: 1px solid #1e293b; }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 10px 24px;
    }
    .stButton > button:hover { opacity: 0.9; }
    
    /* Tables */
    .dataframe { background: #1e293b !important; color: #e2e8f0 !important; }
    
    /* Success/Error messages */
    .stSuccess { background: #065f46; }
    .stError { background: #7f1d1d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.users = {
        "admin": {"password": "admin123", "role": "Admin"},
        "analyst": {"password": "analyst123", "role": "Analyst"},
        "demo": {"password": "demo", "role": "Viewer"}
    }
    st.session_state.loan_results = None
    st.session_state.fraud_results = None

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
BASE = os.path.dirname(__file__)

@st.cache_resource
def load_models():
    m = os.path.join(BASE, "models")
    return {
        "loan_model": joblib.load(f"{m}/loan_model.pkl"),
        "loan_scaler": joblib.load(f"{m}/loan_scaler.pkl"),
        "fraud_model": joblib.load(f"{m}/fraud_model.pkl"),
        "fraud_scaler": joblib.load(f"{m}/fraud_scaler.pkl"),
    }

@st.cache_data
def load_metrics():
    with open(os.path.join(BASE, "models", "metrics.json")) as f:
        return json.load(f)

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def risk_label(prob):
    if prob < 0.3: return "🟢 Low Risk"
    elif prob < 0.6: return "🟡 Medium Risk"
    else: return "🔴 High Risk"

def get_loan_recommendation(prob, row):
    recs = []
    if row.get("missed_payments", 0) > 3:
        recs.append("• Reduce missed payments history before reapplying")
    if row.get("debt_to_income", 0) > 0.4:
        recs.append("• Lower debt-to-income ratio below 40%")
    if row.get("credit_score", 850) < 600:
        recs.append("• Improve credit score by paying bills on time")
    if row.get("employment_years", 5) < 2:
        recs.append("• Build longer employment history (2+ years recommended)")
    if prob > 0.6:
        recs.append("• Strong profile — consider applying for a higher amount")
    if not recs:
        recs.append("• Maintain current financial habits")
    return recs

def get_fraud_recommendation(prob, row):
    recs = []
    if row.get("foreign_txn", 0):
        recs.append("• Verify if this is an authorized international transaction")
    if row.get("distance_from_home", 0) > 100:
        recs.append("• Transaction location is far from home — confirm with customer")
    if row.get("num_txn_today", 0) > 10:
        recs.append("• Unusually high transaction frequency today")
    if prob > 0.7:
        recs.append("• IMMEDIATE REVIEW REQUIRED — Block card pending verification")
    elif prob > 0.4:
        recs.append("• Send SMS verification to customer")
    else:
        recs.append("• Transaction appears legitimate")
    return recs

def clean_loan_data(df):
    required = ['age','income','credit_score','loan_amount','employment_years',
                'debt_to_income','num_credit_lines','missed_payments','education']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    df = df.copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(df['age'].median() if 'age' in df else 35)
    df['income'] = pd.to_numeric(df['income'], errors='coerce').fillna(50000)
    df['credit_score'] = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850).fillna(650)
    df['loan_amount'] = pd.to_numeric(df['loan_amount'], errors='coerce').fillna(10000)
    df['employment_years'] = pd.to_numeric(df['employment_years'], errors='coerce').fillna(3)
    df['debt_to_income'] = pd.to_numeric(df['debt_to_income'], errors='coerce').clip(0, 1).fillna(0.3)
    df['num_credit_lines'] = pd.to_numeric(df['num_credit_lines'], errors='coerce').fillna(3)
    df['missed_payments'] = pd.to_numeric(df['missed_payments'], errors='coerce').fillna(0)
    edu_map = {'high school': 0, 'bachelor': 1, 'master': 2, 'phd': 3}
    df['education_num'] = df['education'].str.lower().map(edu_map).fillna(1)
    return df, None

def clean_fraud_data(df):
    required = ['txn_amount','hour','distance_from_home','avg_daily_spend','num_txn_today','foreign_txn']
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    df = df.copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['foreign_txn'] = df['foreign_txn'].clip(0, 1)
    return df, None

def make_confusion_matrix_fig(cm, title):
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted No','Predicted Yes'],
                yticklabels=['Actual No','Actual Yes'],
                ax=ax, linewidths=0.5, linecolor='#334155')
    ax.set_title(title, color='#e2e8f0', fontsize=11, pad=10)
    for text in ax.texts: text.set_color('white')
    ax.tick_params(colors='#94a3b8', labelsize=8)
    plt.tight_layout()
    return fig

def make_roc_fig(roc_pts, auc_val, title):
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    fprs = [p['fpr'] for p in roc_pts]
    tprs = [p['tpr'] for p in roc_pts]
    ax.plot(fprs, tprs, color='#0ea5e9', lw=2, label=f'AUC = {auc_val:.3f}')
    ax.plot([0,1],[0,1],'--', color='#475569', lw=1)
    ax.set_xlabel('False Positive Rate', color='#94a3b8', fontsize=9)
    ax.set_ylabel('True Positive Rate', color='#94a3b8', fontsize=9)
    ax.set_title(title, color='#e2e8f0', fontsize=11)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_color('#334155')
    ax.spines['right'].set_color('#334155')
    plt.tight_layout()
    return fig

def make_feature_importance_fig(importances, title):
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    labels = [k.replace('_', ' ').title() for k, v in sorted_items]
    values = [v for k, v in sorted_items]
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1e293b')
    ax.set_facecolor('#1e293b')
    colors = ['#0ea5e9','#38bdf8','#7dd3fc','#bae6fd','#8b5cf6','#a78bfa','#c4b5fd','#6ee7b7','#34d399']
    bars = ax.barh(labels[::-1], values[::-1], color=colors[:len(labels)], height=0.6)
    ax.set_xlabel('Importance', color='#94a3b8', fontsize=9)
    ax.set_title(title, color='#e2e8f0', fontsize=11)
    ax.tick_params(colors='#94a3b8', labelsize=8)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────
def login_page():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center; margin-bottom:30px;'>
            <span style='font-size:3rem;'>🏦</span>
            <h1 style='color:#38bdf8; margin:10px 0 5px 0; font-size:1.6rem;'>AI Financial Decision System</h1>
            <p style='color:#64748b; font-size:0.9rem;'>Powered by Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="admin / analyst / demo")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)

            if submitted:
                users = st.session_state.users
                if username in users and users[username]["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = users[username]["role"]
                    st.success(f"Welcome, {username}! ({users[username]['role']})")
                    st.rerun()
                else:
                    st.error("Invalid credentials. Try: admin/admin123")

        st.markdown("""
        <div style='text-align:center; margin-top:20px; color:#475569; font-size:0.8rem;'>
        <b>Demo Accounts:</b> admin/admin123 · analyst/analyst123 · demo/demo
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:16px; background:#1e293b; border-radius:10px; margin-bottom:16px;'>
            <div style='color:#38bdf8; font-weight:700; font-size:1.1rem;'>🏦 FinAI System</div>
            <div style='color:#64748b; font-size:0.8rem; margin-top:4px;'>
                👤 {st.session_state.username} · {st.session_state.role}
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio("📍 Navigation", [
            "📊 Dashboard",
            "💳 Loan Approval",
            "🔐 Fraud Detection",
            "📂 Bulk CSV Upload",
            "👤 Individual Analysis",
            "📈 Model Performance",
            "ℹ️ About"
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown("""
        <div style='color:#475569; font-size:0.75rem; text-align:center;'>
        AI Financial System v1.0<br>
        Built with RandomForest & GradientBoosting<br>
        AUC: 0.93+ (Loan) | 0.99+ (Fraud)
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

        return page

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
def page_dashboard():
    metrics = load_metrics()
    st.markdown("<h1 style='color:#e2e8f0;'>📊 Executive Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:#64748b;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", unsafe_allow_html=True)

    # KPI Row
    cols = st.columns(4)
    kpis = [
        ("2,000", "Training Records", "#38bdf8"),
        (f"{metrics['dataset']['loan_approved_pct']}%", "Loan Approval Rate", "#6ee7b7"),
        (f"{metrics['dataset']['fraud_pct']}%", "Fraud Rate", "#f87171"),
        (f"{metrics['loan']['auc']:.1%}", "Loan Model AUC", "#a78bfa"),
    ]
    for col, (val, label, color) in zip(cols, kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color};'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**🎯 Model Accuracy Comparison**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='#1e293b')
        ax.set_facecolor('#1e293b')
        models = ['Loan\nApproval', 'Fraud\nDetection']
        metrics_list = [
            [metrics['loan']['accuracy'], metrics['loan']['precision'], metrics['loan']['recall'], metrics['loan']['f1']],
            [metrics['fraud']['accuracy'], metrics['fraud']['precision'], metrics['fraud']['recall'], metrics['fraud']['f1']],
        ]
        x = np.arange(4)
        w = 0.3
        labels_m = ['Accuracy', 'Precision', 'Recall', 'F1']
        ax.bar(x - w/2, metrics_list[0], w, color='#0ea5e9', label='Loan Model', alpha=0.9)
        ax.bar(x + w/2, metrics_list[1], w, color='#8b5cf6', label='Fraud Model', alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels_m, color='#94a3b8', fontsize=9)
        ax.set_ylim(0.7, 1.02)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
        ax.set_title('Model Performance Metrics', color='#e2e8f0', fontsize=11)
        for spine in ax.spines.values(): spine.set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown("**🏦 Key Financial Indicators**")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor='#1e293b')
        ax2.set_facecolor('#1e293b')
        categories = ['Model\nAUC (Loan)', 'Model\nAUC (Fraud)', 'Fraud\nDetection%', 'Approval\nAccuracy']
        values_plot = [metrics['loan']['auc'], metrics['fraud']['auc'],
                       metrics['fraud']['recall'], metrics['loan']['accuracy']]
        colors_p = ['#0ea5e9','#8b5cf6','#f87171','#6ee7b7']
        bars = ax2.bar(categories, values_plot, color=colors_p, alpha=0.9, width=0.5)
        for bar, v in zip(bars, values_plot):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', color='#e2e8f0', fontsize=8)
        ax2.set_ylim(0.8, 1.05)
        ax2.tick_params(colors='#94a3b8', labelsize=8)
        ax2.set_title('System Performance Overview', color='#e2e8f0', fontsize=11)
        for spine in ax2.spines.values(): spine.set_color('#334155')
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ROC curves
    c3, c4 = st.columns(2)
    with c3:
        fig3 = make_roc_fig(metrics['loan']['roc_points'], metrics['loan']['auc'], 'Loan Model — ROC Curve')
        st.pyplot(fig3); plt.close()
    with c4:
        fig4 = make_roc_fig(metrics['fraud']['roc_points'], metrics['fraud']['auc'], 'Fraud Model — ROC Curve')
        st.pyplot(fig4); plt.close()


def page_loan():
    models = load_models()
    st.markdown("<h1 style='color:#e2e8f0;'>💳 Loan Approval Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Enter customer details to predict loan approval using Random Forest model</p>", unsafe_allow_html=True)

    with st.form("loan_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 80, 35)
            income = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
            credit_score = st.slider("Credit Score", 300, 850, 680)
        with c2:
            loan_amount = st.number_input("Loan Amount ($)", 1000, 500000, 25000, step=500)
            employment_years = st.number_input("Employment Years", 0, 40, 5)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 0.9, 0.25, 0.01)
        with c3:
            num_credit_lines = st.number_input("Credit Lines", 1, 20, 4)
            missed_payments = st.number_input("Missed Payments (last 2yr)", 0, 20, 0)
            education = st.selectbox("Education", ["High School","Bachelor","Master","PhD"])

        submitted = st.form_submit_button("🔍 Predict Loan Approval", use_container_width=True)

    if submitted:
        edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        features = np.array([[age, income, credit_score, loan_amount, employment_years,
                               debt_to_income, num_credit_lines, missed_payments, edu_map[education]]])
        features_scaled = models['loan_scaler'].transform(features)
        prob = models['loan_model'].predict_proba(features_scaled)[0][1]
        decision = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"
        badge_cls = "badge-approved" if prob >= 0.5 else "badge-rejected"

        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown(f"""
            <div class='metric-card' style='padding:30px;'>
                <div style='font-size:0.9rem; color:#94a3b8; margin-bottom:10px;'>DECISION</div>
                <div style='font-size:1.8rem; font-weight:800; color:{"#6ee7b7" if prob>=0.5 else "#f87171"};'>{decision}</div>
                <div style='margin-top:15px;'>
                    <div style='color:#94a3b8; font-size:0.85rem;'>Approval Probability</div>
                    <div style='font-size:2rem; font-weight:700; color:#38bdf8;'>{prob:.1%}</div>
                </div>
                <div style='margin-top:10px; color:#64748b; font-size:0.8rem;'>{risk_label(1-prob if prob>=0.5 else prob)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown("**📋 AI Recommendations**")
            row = {'missed_payments': missed_payments, 'debt_to_income': debt_to_income,
                   'credit_score': credit_score, 'employment_years': employment_years}
            recs = get_loan_recommendation(prob, row)
            for r in recs:
                st.markdown(f"<div class='info-box'>{r}</div>", unsafe_allow_html=True)

            # Risk gauge bar
            st.markdown("**📊 Risk Score Visualization**")
            fig_g, ax_g = plt.subplots(figsize=(5, 1.5), facecolor='#1e293b')
            ax_g.set_facecolor('#1e293b')
            ax_g.barh(['Approval\nScore'], [prob], color='#0ea5e9', height=0.5)
            ax_g.barh(['Approval\nScore'], [1-prob], left=[prob], color='#334155', height=0.5)
            ax_g.axvline(0.5, color='#f59e0b', lw=2, ls='--', label='Decision Boundary')
            ax_g.set_xlim(0,1); ax_g.tick_params(colors='#94a3b8', labelsize=8)
            ax_g.set_xlabel('Probability', color='#94a3b8', fontsize=8)
            for spine in ax_g.spines.values(): spine.set_color('#334155')
            ax_g.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig_g); plt.close()


def page_fraud():
    models = load_models()
    st.markdown("<h1 style='color:#e2e8f0;'>🔐 Real-Time Fraud Detection</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b;'>Analyze transactions using Gradient Boosting model</p>", unsafe_allow_html=True)

    with st.form("fraud_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            txn_amount = st.number_input("Transaction Amount ($)", 0.01, 50000.0, 250.0, step=10.0)
            hour = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        with c2:
            distance_from_home = st.number_input("Distance from Home (km)", 0.0, 1000.0, 10.0, step=1.0)
            avg_daily_spend = st.number_input("Avg Daily Spend ($)", 10, 5000, 150)
        with c3:
            num_txn_today = st.number_input("# Transactions Today", 1, 50, 3)
            foreign_txn = st.selectbox("International Transaction?", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    if submitted:
        features = np.array([[txn_amount, hour, distance_from_home, avg_daily_spend, num_txn_today, foreign_txn]])
        features_scaled = models['fraud_scaler'].transform(features)
        prob = models['fraud_model'].predict_proba(features_scaled)[0][1]
        is_fraud = prob >= 0.5
        decision = "⚠️ FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE"

        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.markdown(f"""
            <div class='metric-card' style='padding:30px;'>
                <div style='font-size:0.9rem; color:#94a3b8; margin-bottom:10px;'>VERDICT</div>
                <div style='font-size:1.5rem; font-weight:800; color:{"#f87171" if is_fraud else "#6ee7b7"};'>{decision}</div>
                <div style='margin-top:15px;'>
                    <div style='color:#94a3b8; font-size:0.85rem;'>Fraud Probability</div>
                    <div style='font-size:2rem; font-weight:700; color:{"#f87171" if is_fraud else "#38bdf8"};'>{prob:.1%}</div>
                </div>
                <div style='margin-top:10px; color:#64748b; font-size:0.8rem;'>{risk_label(prob)}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            st.markdown("**🔍 Risk Factors**")
            row = {'foreign_txn': foreign_txn, 'distance_from_home': distance_from_home,
                   'num_txn_today': num_txn_today}
            recs = get_fraud_recommendation(prob, row)
            for r in recs:
                color = "#7f1d1d" if is_fraud else "#064e3b"
                st.markdown(f"<div class='info-box' style='border-color:{"#f87171" if is_fraud else "#6ee7b7"};'>{r}</div>", unsafe_allow_html=True)

            # Speedometer-style chart
            fig_s, ax_s = plt.subplots(figsize=(5, 2), facecolor='#1e293b')
            ax_s.set_facecolor('#1e293b')
            gradient = np.linspace(0, 1, 100).reshape(1, -1)
            ax_s.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 1, 0, 1])
            ax_s.axvline(prob, color='white', lw=3, label=f'Score: {prob:.2f}')
            ax_s.set_yticks([]); ax_s.set_xlabel('Fraud Score (0=Safe, 1=Fraud)', color='#94a3b8', fontsize=9)
            ax_s.tick_params(colors='#94a3b8', labelsize=8)
            ax_s.set_title('Transaction Risk Meter', color='#e2e8f0', fontsize=11)
            ax_s.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_s); plt.close()


def page_bulk_upload():
    models = load_models()
    st.markdown("<h1 style='color:#e2e8f0;'>📂 Bulk CSV Upload & Prediction</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💳 Loan Predictions", "🔐 Fraud Predictions"])

    with tab1:
        st.markdown("""
        <div class='info-box'>
        <b>Required columns:</b> age, income, credit_score, loan_amount, employment_years,
        debt_to_income, num_credit_lines, missed_payments, education
        </div>
        """, unsafe_allow_html=True)

        # Download template
        sample_loan = pd.read_csv(os.path.join(BASE, 'data', 'sample_loan_data.csv')).head(5)
        csv_bytes = sample_loan.to_csv(index=False).encode()
        st.download_button("📥 Download Sample Loan CSV", csv_bytes, "sample_loan.csv", "text/csv")

        uploaded = st.file_uploader("Upload Loan CSV", type=['csv'], key='loan_upload')
        if uploaded:
            df = pd.read_csv(uploaded)
            st.markdown(f"**Uploaded:** {len(df)} rows, {len(df.columns)} columns")
            st.dataframe(df.head(5), use_container_width=True)

            # Clean
            with st.spinner("🧹 Cleaning data..."):
                cleaned, err = clean_loan_data(df)
            if err:
                st.error(f"Data issue: {err}")
            else:
                st.success("✅ Data cleaned successfully")
                with st.spinner("🤖 Running predictions..."):
                    features = cleaned[['age','income','credit_score','loan_amount','employment_years',
                                        'debt_to_income','num_credit_lines','missed_payments','education_num']]
                    scaled = models['loan_scaler'].transform(features)
                    probs = models['loan_model'].predict_proba(scaled)[:,1]
                    cleaned['approval_probability'] = probs.round(4)
                    cleaned['decision'] = ['APPROVED' if p >= 0.5 else 'REJECTED' for p in probs]
                    cleaned['risk_level'] = [risk_label(1-p if p>=0.5 else p) for p in probs]

                st.markdown("**📊 Results Preview**")
                result_cols = ['age','income','credit_score','loan_amount','approval_probability','decision','risk_level']
                st.dataframe(cleaned[[c for c in result_cols if c in cleaned.columns]].head(20), use_container_width=True)

                # Summary chart
                fig_pie, ax_pie = plt.subplots(figsize=(4, 3.5), facecolor='#1e293b')
                ax_pie.set_facecolor('#1e293b')
                counts = cleaned['decision'].value_counts()
                ax_pie.pie(counts.values, labels=counts.index,
                           colors=['#6ee7b7','#f87171'], autopct='%1.1f%%',
                           textprops={'color':'#e2e8f0', 'fontsize':9})
                ax_pie.set_title('Approval Distribution', color='#e2e8f0', fontsize=11)
                st.pyplot(fig_pie); plt.close()

                # Download results
                result_csv = cleaned.to_csv(index=False).encode()
                st.download_button("📥 Download Results CSV", result_csv, "loan_predictions.csv", "text/csv")
                st.session_state.loan_results = cleaned

    with tab2:
        st.markdown("""
        <div class='info-box'>
        <b>Required columns:</b> txn_amount, hour, distance_from_home, avg_daily_spend, num_txn_today, foreign_txn
        </div>
        """, unsafe_allow_html=True)

        sample_fraud = pd.read_csv(os.path.join(BASE, 'data', 'sample_fraud_data.csv')).head(5)
        csv_bytes2 = sample_fraud.to_csv(index=False).encode()
        st.download_button("📥 Download Sample Fraud CSV", csv_bytes2, "sample_fraud.csv", "text/csv")

        uploaded2 = st.file_uploader("Upload Transaction CSV", type=['csv'], key='fraud_upload')
        if uploaded2:
            df2 = pd.read_csv(uploaded2)
            st.markdown(f"**Uploaded:** {len(df2)} rows")
            st.dataframe(df2.head(5), use_container_width=True)

            cleaned2, err2 = clean_fraud_data(df2)
            if err2:
                st.error(f"Data issue: {err2}")
            else:
                st.success("✅ Data cleaned")
                features2 = cleaned2[['txn_amount','hour','distance_from_home',
                                      'avg_daily_spend','num_txn_today','foreign_txn']]
                scaled2 = models['fraud_scaler'].transform(features2)
                probs2 = models['fraud_model'].predict_proba(scaled2)[:,1]
                cleaned2['fraud_probability'] = probs2.round(4)
                cleaned2['verdict'] = ['FRAUD' if p >= 0.5 else 'LEGITIMATE' for p in probs2]
                cleaned2['risk_level'] = [risk_label(p) for p in probs2]

                st.dataframe(cleaned2.head(20), use_container_width=True)
                result_csv2 = cleaned2.to_csv(index=False).encode()
                st.download_button("📥 Download Results CSV", result_csv2, "fraud_predictions.csv", "text/csv")
                st.session_state.fraud_results = cleaned2


def page_individual():
    st.markdown("<h1 style='color:#e2e8f0;'>👤 Individual Customer Risk Analysis</h1>", unsafe_allow_html=True)
    models = load_models()

    customer_id = st.text_input("Customer ID / Name", "CUST-2024-001")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**💳 Loan Profile**")
        age = st.number_input("Age", 18, 80, 34, key='ind_age')
        income = st.number_input("Income ($)", 10000, 500000, 72000, key='ind_inc')
        credit_score = st.number_input("Credit Score", 300, 850, 720, key='ind_cs')
        loan_amount = st.number_input("Loan Request ($)", 1000, 500000, 35000, key='ind_la')
        employment_years = st.number_input("Employment Years", 0, 40, 8, key='ind_ey')
        debt_to_income = st.slider("DTI Ratio", 0.0, 0.9, 0.22, key='ind_dti')
        num_credit_lines = st.number_input("Credit Lines", 1, 20, 5, key='ind_ncl')
        missed_payments = st.number_input("Missed Payments", 0, 20, 1, key='ind_mp')
        education = st.selectbox("Education", ["High School","Bachelor","Master","PhD"], key='ind_edu')

    with c2:
        st.markdown("**🔐 Recent Transaction**")
        txn_amount = st.number_input("Txn Amount ($)", 0.01, 50000.0, 890.0, key='ind_ta')
        hour = st.slider("Hour", 0, 23, 22, key='ind_hr')
        distance = st.number_input("Distance from Home (km)", 0.0, 1000.0, 45.0, key='ind_dist')
        avg_spend = st.number_input("Avg Daily Spend ($)", 10, 5000, 200, key='ind_as')
        num_txn = st.number_input("# Txn Today", 1, 50, 7, key='ind_ntxn')
        foreign = st.selectbox("International?", [0, 1], key='ind_fgn', format_func=lambda x: "Yes" if x else "No")

    if st.button("🔍 Run Full Risk Analysis", use_container_width=True):
        # Loan prediction
        edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        loan_feat = np.array([[age, income, credit_score, loan_amount, employment_years,
                               debt_to_income, num_credit_lines, missed_payments, edu_map[education]]])
        loan_prob = models['loan_model'].predict_proba(models['loan_scaler'].transform(loan_feat))[0][1]

        # Fraud prediction
        fraud_feat = np.array([[txn_amount, hour, distance, avg_spend, num_txn, foreign]])
        fraud_prob = models['fraud_model'].predict_proba(models['fraud_scaler'].transform(fraud_feat))[0][1]

        overall_risk = (fraud_prob * 0.6 + (1-loan_prob) * 0.4)

        st.markdown("---")
        st.markdown(f"## 📋 Risk Report: {customer_id}")

        col1, col2, col3 = st.columns(3)
        col1.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{"#6ee7b7" if loan_prob>=0.5 else "#f87171"};'>
                {"✅ Approved" if loan_prob>=0.5 else "❌ Rejected"}
            </div>
            <div class='metric-label'>Loan Decision ({loan_prob:.1%})</div>
        </div>""", unsafe_allow_html=True)

        col2.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{"#f87171" if fraud_prob>=0.5 else "#6ee7b7"};'>
                {"⚠️ Fraud" if fraud_prob>=0.5 else "✅ Legit"}
            </div>
            <div class='metric-label'>Txn Verdict ({fraud_prob:.1%})</div>
        </div>""", unsafe_allow_html=True)

        col3.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{"#f87171" if overall_risk>0.5 else "#6ee7b7"};'>
                {overall_risk:.1%}
            </div>
            <div class='metric-label'>Overall Risk Score</div>
        </div>""", unsafe_allow_html=True)

        # Spider chart
        st.markdown("**📊 Risk Factor Breakdown**")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3), facecolor='#1e293b')
        ax_bar.set_facecolor('#1e293b')
        factors = ['Credit\nScore Risk', 'Debt\nRatio', 'Payment\nHistory', 'Fraud\nScore', 'Distance\nRisk']
        scores = [
            1 - (credit_score-300)/550,
            debt_to_income,
            min(missed_payments/10, 1),
            fraud_prob,
            min(distance/500, 1)
        ]
        colors_bar = ['#0ea5e9' if s < 0.4 else '#f59e0b' if s < 0.7 else '#f87171' for s in scores]
        ax_bar.bar(factors, scores, color=colors_bar, alpha=0.9, width=0.5)
        ax_bar.axhline(0.5, color='#f59e0b', ls='--', lw=1, alpha=0.7, label='Risk Threshold')
        ax_bar.set_ylim(0, 1); ax_bar.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax_bar.spines.values(): spine.set_color('#334155')
        ax_bar.set_ylabel('Risk Score', color='#94a3b8', fontsize=9)
        ax_bar.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig_bar); plt.close()

        # AI Summary
        st.markdown("**💡 AI Summary & Recommendations**")
        summary = f"""
        **Customer {customer_id} — Risk Assessment Summary**
        
        🏦 **Loan Assessment:** The customer has a {loan_prob:.1%} probability of loan repayment.
        {"✅ Loan APPROVED — Strong financial profile." if loan_prob >= 0.5 else "❌ Loan REJECTED — Risk factors present."}
        
        🔐 **Transaction Security:** The analyzed transaction has a {fraud_prob:.1%} fraud probability.
        {"⚠️ SUSPICIOUS — Recommend immediate verification." if fraud_prob >= 0.5 else "✅ LEGITIMATE — Transaction appears normal."}
        
        📊 **Overall Profile:** {risk_label(overall_risk)}
        """
        st.markdown(summary)


def page_model_performance():
    metrics = load_metrics()
    st.markdown("<h1 style='color:#e2e8f0;'>📈 Model Performance Analytics</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["💳 Loan Model", "🔐 Fraud Model"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        for col, (k, v) in zip([c1,c2,c3,c4], [
            ("AUC", metrics['loan']['auc']), ("Accuracy", metrics['loan']['accuracy']),
            ("Precision", metrics['loan']['precision']), ("F1 Score", metrics['loan']['f1'])
        ]):
            col.markdown(f"<div class='metric-card'><div class='metric-value'>{v:.3f}</div><div class='metric-label'>{k}</div></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = make_confusion_matrix_fig(metrics['loan']['confusion_matrix'], 'Loan Confusion Matrix')
            st.pyplot(fig); plt.close()
        with c2:
            fig = make_roc_fig(metrics['loan']['roc_points'], metrics['loan']['auc'], 'Loan ROC Curve')
            st.pyplot(fig); plt.close()
        with c3:
            fig = make_feature_importance_fig(metrics['loan']['feature_importances'], 'Feature Importances')
            st.pyplot(fig); plt.close()

        st.markdown("""
        <div class='info-box'>
        <b>Algorithm:</b> Random Forest (100 estimators, max depth 8) <br>
        <b>Training:</b> 80/20 train-test split, StandardScaler normalization<br>
        <b>Best Use:</b> Credit scoring and loan approval with interpretable risk factors
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        c1, c2, c3, c4 = st.columns(4)
        for col, (k, v) in zip([c1,c2,c3,c4], [
            ("AUC", metrics['fraud']['auc']), ("Accuracy", metrics['fraud']['accuracy']),
            ("Precision", metrics['fraud']['precision']), ("F1 Score", metrics['fraud']['f1'])
        ]):
            col.markdown(f"<div class='metric-card'><div class='metric-value'>{v:.3f}</div><div class='metric-label'>{k}</div></div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig = make_confusion_matrix_fig(metrics['fraud']['confusion_matrix'], 'Fraud Confusion Matrix')
            st.pyplot(fig); plt.close()
        with c2:
            fig = make_roc_fig(metrics['fraud']['roc_points'], metrics['fraud']['auc'], 'Fraud ROC Curve')
            st.pyplot(fig); plt.close()
        with c3:
            fig = make_feature_importance_fig(metrics['fraud']['feature_importances'], 'Feature Importances')
            st.pyplot(fig); plt.close()

        st.markdown("""
        <div class='info-box'>
        <b>Algorithm:</b> Gradient Boosting Classifier (100 estimators, max depth 5)<br>
        <b>Training:</b> 80/20 train-test split, StandardScaler normalization<br>
        <b>Best Use:</b> Real-time anomaly detection in financial transactions
        </div>
        """, unsafe_allow_html=True)


def page_about():
    st.markdown("<h1 style='color:#e2e8f0;'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#1e293b; padding:30px; border-radius:12px; border:1px solid #334155;'>
    <h2 style='color:#38bdf8;'>🏦 AI-Powered Financial Decision System</h2>
    <p style='color:#94a3b8;'>Final Year Project | B.Sc Artificial Intelligence & Data Science</p>
    <p style='color:#94a3b8;'>KPR College of Arts, Science and Research</p>
    
    <hr style='border-color:#334155; margin:20px 0;'>
    
    <h3 style='color:#e2e8f0;'>📚 Based On</h3>
    <p style='color:#94a3b8;'>Chapter: "Artificial Intelligence and Machine Learning in Financial Decision Systems"<br>
    Published in International Edited Volume (2024)</p>
    
    <hr style='border-color:#334155; margin:20px 0;'>
    
    <h3 style='color:#e2e8f0;'>🤖 ML Models Used</h3>
    <table style='width:100%; color:#94a3b8;'>
    <tr><th style='color:#38bdf8; text-align:left;'>Model</th><th style='color:#38bdf8;'>Task</th><th style='color:#38bdf8;'>AUC</th></tr>
    <tr><td>Random Forest (100 trees)</td><td>Loan Approval</td><td>0.932+</td></tr>
    <tr><td>Gradient Boosting (100 trees)</td><td>Fraud Detection</td><td>0.997+</td></tr>
    </table>
    
    <hr style='border-color:#334155; margin:20px 0;'>
    
    <h3 style='color:#e2e8f0;'>✨ Key Features</h3>
    <ul style='color:#94a3b8;'>
        <li>🔐 Secure login & role-based access (Admin / Analyst / Viewer)</li>
        <li>📂 CSV bulk upload with automatic data cleaning</li>
        <li>🤖 Real-time loan approval prediction</li>
        <li>💳 Transaction fraud detection with risk scoring</li>
        <li>📊 Interactive charts: ROC, Confusion Matrix, Feature Importance</li>
        <li>👤 Individual customer full risk profile</li>
        <li>💡 AI-generated recommendations for each decision</li>
        <li>📥 Downloadable prediction results as CSV</li>
        <li>💾 Pre-trained .pkl models (saved via joblib)</li>
    </ul>
    
    <hr style='border-color:#334155; margin:20px 0;'>
    
    <h3 style='color:#e2e8f0;'>📖 References</h3>
    <ul style='color:#64748b; font-size:0.85rem;'>
        <li>Altman (1968) — Credit scoring discriminant analysis</li>
        <li>Chen & Guestrin (2016) — XGBoost gradient boosting</li>
        <li>Fischer & Krauss (2018) — LSTM for financial prediction</li>
        <li>Lundberg & Lee (2017) — SHAP explainability values</li>
        <li>World Bank (2024) — AI in financial inclusion</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    page = render_sidebar()

    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "💳 Loan Approval":
        page_loan()
    elif page == "🔐 Fraud Detection":
        page_fraud()
    elif page == "📂 Bulk CSV Upload":
        page_bulk_upload()
    elif page == "👤 Individual Analysis":
        page_individual()
    elif page == "📈 Model Performance":
        page_model_performance()
    elif page == "ℹ️ About":
        page_about()

if __name__ == "__main__":
    main()
