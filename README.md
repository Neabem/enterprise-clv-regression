# 🏢 Enterprise Customer Lifetime Value (CLV) & Revenue Forecasting

An end-to-end Machine Learning regression and explainability pipeline engineered to forecast corporate annual contract value and customer expansion spend.

This repository demonstrates a complete enterprise workflow: extracting and aggregating records across a 5-table relational SQLite schema via Common Table Expressions (CTEs), conducting visual exploratory data analysis, preventing target leakage with a Scikit-Learn `ColumnTransformer` (incorporating `RobustScaler` and `SimpleImputer`), benchmarking linear and regularized estimators (Ridge $L_2$, Lasso $L_1$) against gradient boosted decision trees (`XGBRegressor`), executing 5-fold cross-validated hyperparameter optimization (`RandomizedSearchCV`), diagnosing sample efficiency with empirical Learning Curves, executing rigorous residual error diagnostics ($y=x$ and $e=0$), interpreting predictions via SHAP (Shapley Additive exPlanations), and deploying an interactive web application (`clv_app.py`) with Streamlit.

---

## 🛠️ Tech Stack & Tooling

* **Relational Database / SQL:** SQLite3 (CTEs, Multi-Table Inner Joins across 5 relational entities)
* **Data Manipulation & Preprocessing:** Pandas, NumPy, Scikit-Learn (`Pipeline`, `ColumnTransformer`, `RobustScaler`, `SimpleImputer`, `OneHotEncoder`)
* **Data Visualization & Diagnostics:** Matplotlib, Seaborn (Masked Pearson heatmaps `np.triu`, Bivariate Regressions, Distribution Skewness)
* **Regression Modeling & Benchmarking:** Scikit-Learn (`LinearRegression`, `Ridge`, `Lasso`), XGBoost (`XGBRegressor`)
* **Explainable AI (XAI):** SHAP (`TreeExplainer`, Global Beeswarm plots, Local Waterfall plots)
* **Production Deployment & Serialization:** `joblib`, Streamlit (`clv_app.py`, `@st.cache_resource`, real-time inference)

---

## 🏗️ Relational Schema (`enterprise_clv.db`)

The analytical dataset is partitioned across 5 normalized relational tables representing 2,500 corporate accounts:
1. `accounts`: Primary key `account_id`, `company_name`, `industry`, `company_size`, `region`, `contract_tier`, `created_date`.
2. `contract_metrics`: Account tenure (`tenure_months`), allocated `seat_licenses`, baseline annual recurring revenue (`base_arr`), negotiated `discount_pct`, `payment_terms`, and dedicated account manager flag (`has_dedicated_success_manager`).
3. `product_telemetry`: Monthly active users (`monthly_active_users`), monthly API utilization (`api_call_volume_thousands`), cloud storage usage (`storage_gb`), platform feature score (`feature_adoption_score`), and active tool integrations count (`integrations_count`).
4. `support_and_satisfaction`: Historical support burden (`support_tickets_count`), mean time to resolution (`avg_resolution_hours`), manager escalations (`escalations_count`), client survey rating (`nps_score`), and open tickets (`unresolved_tickets`).
5. `clv_targets`: Target continuous outcome variable (`target_annual_clv_spend`).

---

## 📊 Regression Benchmark Leaderboard

| Model Architecture | Regularization | Test $R^2$ | Test MAE (\$) | Test RMSE (\$) | Test MAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ordinary Least Squares (OLS)** | None | 0.9947 | \$4,331.98 | \$6,157.95 | 13.60% |
| **Ridge Regression** | $L_2$ Penalty ($\alpha=10.0$) | 0.9945 | \$4,375.10 | \$6,242.38 | 13.60% |
| **Lasso Regression** | $L_1$ Penalty ($\alpha=5.0$) | 0.9947 | \$4,324.75 | \$6,152.94 | 13.52% |
| **XGBoost Regressor (Champion)** | Tree Ensemble (Depth=4, $\eta=0.08$) | **0.9922** | **\$4,947.09** | **\$7,425.20** | **12.40%** |

---

## 🔍 Model Explainability & Residual Diagnostics

* **Residual Diagnostics:** Evaluated on an unseen 20% test partition. Actual vs. Predicted scatter strictly follows the ideal $y = x$ trajectory across both mid-market and enterprise VIP tiers. Residuals ($e = y - \hat{y}$) are symmetrically centered around \$0 with no heteroscedastic fanning, proving uniform error variance across all account scales.
* **SHAP Global Drivers:** Feature attribution reveals that baseline contract magnitude (`base_arr`), monthly API call volume (`api_call_volume_thousands`), and dedicated CSM support provide the highest positive dollar lift. Conversely, high ticket escalations and steep discount percentages apply immediate negative penalties.

---

## 🚀 Interactive Streamlit Application (`clv_app.py`)

The production pipeline is serialized using `joblib` into `enterprise_clv_pipeline.joblib`. An interactive web application provides enterprise account managers with real-time value scoring and local explainability:

```bash
# Launch the live application
cd enterprise-clv-regression
streamlit run clv_app.py
```

### Key UI Features:
* **Interactive Telemetry Sliders:** Adjust seat count, API consumption, storage, and platform adoption.
* **Cached In-Memory Inference:** Zero-latency scoring via `@st.cache_resource`.
* **Embedded SHAP Waterfall Plot:** Visualizes the exact positive and negative dollar drivers pushing that specific account's valuation above or below baseline expected revenue.

---

## 👤 Author & Credentials
* **Engineer:** Omosaiye Nehemiah
* **Degree:** BTech in Mathematics (4.30/5.00 CGPA, Second Class Upper) | Federal University of Technology Owerri (FUTO)
* **GitHub:** [@Neabem](https://github.com/Neabem)
* **LinkedIn:** [linkedin.com/in/omosaiye-nehemiah-783a44288](https://www.linkedin.com/in/omosaiye-nehemiah-783a44288?utm_source=share_via&utm_content=profile&utm_medium=member_ios)
