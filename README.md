# 🎯 ZeroC-AI — Zero Customer Complaint Analytics & Prediction System (PS-036 · ABT-CPD)

A complete, unified AI-powered platform for customer complaint intelligence, risk prediction, root-cause diagnosis, action recommendation, and executive monitoring.

This application **unifies and merges all features** from both the **ZeroC-AI Prototype Dashboard** and the **ZeroC-AI Streamlit Intelligence Platform** into a single cohesive, production-ready system.

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone or navigate to the repository
cd zeroc-ai

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Platform
```bash
streamlit run app.py
```
Or use the launcher scripts:
- **Windows**: Double-click `run.bat`
- **Linux / macOS**: Execute `./run.sh`

### 3. Run Self-Tests & Smoke Verification
```bash
python test_smoke.py
```

---

## 📋 Comprehensive Coverage of the 9 Problem-Statement Deliverables

| # | Deliverable | Page in Unified App | Key Features & Implementation |
|---|---|---|---|
| **1** | **Classify complaints** | *Classify & Log* | Hierarchical NLP (Category + Sub-defect) using TF-IDF and balanced Logistic Regression. Confidence scoring, low-confidence warning (<50%), transparent term-level explanations, similar past case retrieval, and human-in-the-loop active learning queue. Optional Sentence-BERT upgrade via `ZEROC_SBERT=1`. |
| **2** | **Pareto analysis** | *Command Centre*, *Pareto & History* | 80-20 Pareto analysis across multiple dimensions (Category, Product, Customer, Line, 6M Root Cause, Customer Type) with interactive year-range slider (2008–2026) and dual-axis chart highlighting the vital few in orange/red. |
| **3** | **Recurring & chronic problems** | *Patterns & Root Cause* | Rolling recurrence counter (same defect within 6 months), chronic offender detector table (≥4 complaints in 12m or ≥10 in 24m) with status filters (`Chronic`, `Recurring`, `Occasional`), repeat CAPA failure tracking, and monthly historical timeline with effective action markers. |
| **4** | **Relationship mapping** | *Patterns & Root Cause* | Cross-dimensional lift matrices (Product, Customer, Line vs Category), Apriori association rules mining (Support, Confidence, Lift), 6M Ishikawa classification (Man, Machine, Material, Method, Measurement, Environment), and batch sensor parameter distribution box plots. |
| **5** | **High-risk products & processes** | *Command Centre*, *Risk & Prediction* | Multi-factor composite risk scoring (0–100) combining frequency, severity, recency, trend, and recurrence with RAG tiers (RED/AMBER/GREEN). Product and customer risk league tables. |
| **6** | **Predict complaint risk** | *Risk & Prediction* | XGBoost classifiers with monotonic constraints predicting 30/60/90-day probability of complaint, Platt calibration, time-based holdout validation, SHAP explainability waterfall/bars, and **interactive pre-dispatch batch risk simulator** with real-time clearance decisions (`HOLD LOT` / `ENHANCED CHECKS` / `RELEASE`). |
| **7** | **Recommend preventive actions** | *Preventive Recommendations & CAPA* | Wilson score lower bound effectiveness ranking on historical CAPA outcomes, similar case retrieval (cosine similarity), automated SOP/HACCP rule checks, and interactive **CAPA creation workflow & tracking table**. |
| **8** | **Management dashboard & alerts** | *Command Centre*, *Data, SQL & Integration* | Executive KPI cards with delta comparisons, SPC early warning (c-chart, Western Electric 7-run rule, CUSUM alarms), dynamic threshold slider, automated email notification via SMTP, and local `/outbox` digest. |
| **9** | **KPIs towards zero** | *KPIs: Road to Zero* | Strategic scorecard (PPM, repeat rate, CAPA effectiveness, closure time, 36-month glide path to zero, early-warning backtest hit rates, and longest complaint-free streaks by product-category pair). |

---

## 📂 Project Architecture

```
zeroc-ai/
├── app.py                      # Master Streamlit executive and operational dashboard
├── load_real_data.py           # CLI importer for external real ABT-CPD Excel/CSV files
├── test_smoke.py               # Automated smoke test suite
├── requirements.txt            # Python dependencies
├── run.bat / run.sh            # One-click startup scripts
├── .streamlit/
│   └── config.toml             # Custom theme configuration
├── data/
│   ├── zeroc.db                # SQLite relational database (19 years, ~2,800 complaints, ~8,000 batches)
│   └── complaints.db           # Bundled standalone database
├── models/
│   ├── classifier.joblib       # Serialized NLP hierarchical classifier
│   ├── pair_models.joblib      # Calibrated XGBoost pair risk models (p30/p60/p90)
│   ├── batch_models.joblib     # XGBoost pre-dispatch batch clearance models
│   ├── nlp_metrics.json        # Test-set classification metrics
│   └── registry.json           # Versioned model registry
├── powerbi/
│   ├── ZeroC_AI_PowerBI_Model.xlsx # Complete Power BI ready Excel model
│   ├── PowerBI_Guide.md        # Comprehensive DAX measures and dashboard build guide
│   ├── dim_product.csv         # Product dimension table
│   ├── dim_customer.csv        # Customer dimension table
│   ├── dim_date.csv            # Date dimension table
│   ├── fact_complaints.csv     # Fact complaints table
│   ├── fact_batches.csv        # Fact batches with sensor readings
│   ├── fact_alerts.csv         # Fact alerts table
│   ├── fact_chronic.csv        # Fact chronic offenders table
│   ├── fact_kpi.csv            # Fact KPI targets table
│   └── fact_risk_forecast.csv  # Fact predicted risk table
├── sql/
│   ├── schema.sql              # Relational DDL schema & v_master view
│   └── queries.sql             # Ready-to-run analytical SQL queries
└── zeroc/
    ├── __init__.py
    ├── alerts.py               # Dynamic threshold alert engine & SMTP notification
    ├── analytics.py            # Pareto, SPC, lift matrices, recurrence, streaks & KPIs
    ├── config.py               # Constants, taxonomy, thresholds & paths
    ├── db.py                   # Database connection and CRUD helpers
    ├── export_powerbi.py       # Star-schema CSV & Excel exporter
    ├── generate_data.py        # 19-year realistic relational dataset generator
    ├── model.py                # XGBoost pair & batch models, Platt scaling & SHAP
    ├── nlp.py                  # Hierarchical text classification & theme clustering
    ├── pipeline.py             # End-to-end retraining & ingestion pipeline
    └── recommender.py          # Wilson lower bound action ranking & SOP rules
```

---

## 🔌 Using Real ABT-CPD Data

To ingest real factory and complaint records:
1. Open `load_real_data.py` and adjust `COLUMN_MAP` if your Excel/CSV headers differ.
2. Run the loader:
   ```bash
   python load_real_data.py path/to/complaints.xlsx [batches_or_volumes.csv]
   ```
3. Set the environment variable:
   ```bash
   # Windows:
   set ZC_DATA=real
   # Linux / macOS:
   export ZC_DATA=real
   ```
4. Start the dashboard or click **"🔄 Retrain models now"** in the sidebar.

---

## 📊 Power BI Integration
- The `/powerbi` directory contains a pre-built star-schema ready to import into Power BI Desktop.
- Follow `powerbi/PowerBI_Guide.md` for copy-paste DAX formulas (PPM, Repeat Rate %, Pareto Cum %, SPC UCL limits, Risk Forecast measures).
- You can also export refreshed files directly from the dashboard on the **Data, SQL & Integration** tab.
