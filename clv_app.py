import os
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Enterprise CLV Revenue Forensics", layout="wide", page_icon="🏢")

st.title("🏢 Enterprise Customer Lifetime Value (CLV) & Revenue Forensics")
st.markdown("Real-time corporate contract forecasting, multi-model regression & SHAP explainability.")

st.sidebar.header("📊 Corporate Account Telemetry")

seat_licenses = st.sidebar.number_input("Seat Licenses", min_value=5, max_value=2000, value=120, step=10)
tenure_months = st.sidebar.slider("Account Tenure (Months)", min_value=1, max_value=60, value=24, step=1)
api_calls_k = st.sidebar.number_input("Monthly API Volume (Thousands)", min_value=10.0, max_value=50000.0, value=1250.0, step=50.0)
storage_gb = st.sidebar.slider("Cloud Storage (GB)", min_value=20.0, max_value=80000.0, value=3500.0, step=100.0)
feature_adoption = st.sidebar.slider("Feature Adoption Score (1-100)", min_value=10.0, max_value=100.0, value=75.0, step=1.0)
discount_pct = st.sidebar.slider("Negotiated Discount %", min_value=0.0, max_value=35.0, value=10.0, step=0.5)

company_size = st.sidebar.selectbox("Company Size", ["Startup", "Mid-Market", "Enterprise"])
contract_tier = st.sidebar.selectbox("Contract Tier", ["Standard", "Professional", "Enterprise VIP"])
industry = st.sidebar.selectbox("Industry", ["Fintech", "Healthcare", "E-Commerce", "SaaS", "Manufacturing", "EdTech"])
region = st.sidebar.selectbox("Region", ["North America", "Europe", "Asia-Pacific", "Latin America", "Africa"])
payment_terms = st.sidebar.selectbox("Payment Terms", ["Monthly", "Quarterly", "Annual Upfront"])
has_csm = 1 if st.sidebar.checkbox("Dedicated Success Manager Assigned", value=True) else 0

# Baseline ARR calculation approximation
base_arr = float(seat_licenses * 160.0 + 5000.0)

@st.cache_resource
def load_clv_bundle():
    paths = ['enterprise_clv_pipeline.joblib', 'enterprise-clv-regression/enterprise_clv_pipeline.joblib']
    for p in paths:
        if os.path.exists(p):
            bundle = joblib.load(p)
            explainer = shap.TreeExplainer(bundle['model'])
            return bundle, explainer
    return None, None

bundle, explainer = load_clv_bundle()

if bundle is None:
    st.warning("⚠️ Model bundle `enterprise_clv_pipeline.joblib` not found yet. Complete Part 7 in `Capstone_4_Enterprise_CLV_Regression.ipynb` to serialize your trained model!")
else:
    # Assemble raw payload
    raw_dict = {
        'industry': industry,
        'company_size': company_size,
        'region': region,
        'contract_tier': contract_tier,
        'tenure_months': tenure_months,
        'seat_licenses': seat_licenses,
        'base_arr': base_arr,
        'discount_pct': discount_pct,
        'payment_terms': payment_terms,
        'has_dedicated_success_manager': has_csm,
        'monthly_active_users': int(seat_licenses * 0.95),
        'api_call_volume_thousands': api_calls_k,
        'storage_gb': storage_gb,
        'feature_adoption_score': feature_adoption,
        'integrations_count': 5,
        'support_tickets_count': 6,
        'avg_resolution_hours': 12.0,
        'escalations_count': 0,
        'nps_score': 8.5,
        'unresolved_tickets': 0
    }
    input_df = pd.DataFrame([raw_dict])
    
    # Transform and Predict
    trans = bundle['preprocessor'].transform(input_df)
    trans_df = pd.DataFrame(trans, columns=bundle['encoded_features'])
    predicted_clv = float(bundle['model'].predict(trans_df)[0])
    
    # 3-Column Executive Scorecard
    st.subheader("📈 Annual Contract Valuation Scorecard")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.metric(label="Forecasted Annual CLV Spend", value=f"${predicted_clv:,.2f}")
    with c2:
        expansion_pct = ((predicted_clv - base_arr) / base_arr) * 100.0
        st.metric(label="Net ARR Expansion", value=f"{expansion_pct:+.1f}%", delta=f"${predicted_clv - base_arr:,.2f}")
    with c3:
        status = "💎 High-Growth VIP Account" if predicted_clv >= 100000 else "⭐ Core Growth Account"
        st.info(status)
        
    # Local SHAP Waterfall
    st.subheader("🔍 Local Revenue Drivers & Feature Forensics")
    shap_vals = explainer(trans_df)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    shap.plots.waterfall(shap_vals[0], show=False)
    plt.title(f"SHAP Waterfall: Predicted Valuation = ${predicted_clv:,.2f}", fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
