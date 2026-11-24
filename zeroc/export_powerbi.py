"""Export a star schema for Power BI (CSV + one Excel workbook) and write the Power BI build guide with DAX measures."""
import pandas as pd
from . import db, analytics as A
from .config import *

def export():
    PBI_DIR.mkdir(exist_ok=True); m = A.add_recurrence(db.master()); b = db.batches(); d = {}
    d["fact_complaints"] = m.drop(columns=["t"]).assign(is_repeat=(m.repeat_count > 0).astype(int), severity=m.category.map(SEVERITY))
    d["fact_batches"] = b.drop(columns=["t"]); d["dim_product"] = db.read("SELECT * FROM products"); d["dim_customer"] = db.read("SELECT * FROM customers")
    d["dim_date"] = pd.DataFrame({"date": pd.date_range(START, TODAY)}).assign(year=lambda x: x.date.dt.year, month=lambda x: x.date.dt.month, ym=lambda x: x.date.dt.strftime("%Y-%m"))
    d["fact_risk_forecast"] = db.read("SELECT * FROM predictions"); d["fact_alerts"] = db.read("SELECT * FROM alerts"); d["fact_product_risk_score"] = A.entity_risk(m, "product")
    d["fact_customer_risk_score"] = A.entity_risk(m, "customer"); d["fact_kpi"] = A.kpis(m, b, db.read("SELECT * FROM backtest")); d["fact_chronic"] = A.chronic(m)
    with pd.ExcelWriter(PBI_DIR / "ZeroC_AI_PowerBI_Model.xlsx") as xw:
        for k, v in d.items(): v.to_csv(PBI_DIR / f"{k}.csv", index=False); v.head(100000).to_excel(xw, sheet_name=k[:31], index=False)
    (PBI_DIR / "PowerBI_Guide.md").write_text(GUIDE); return list(d)

GUIDE = '''# ZeroC-AI · Power BI build guide
Power BI is the enterprise reporting layer; the Python pipeline refreshes the same tables nightly.

## 1. Connect (Get Data)
* **Quick start:** Get Data → Excel → `ZeroC_AI_PowerBI_Model.xlsx` (or Folder → this `powerbi/` folder of CSVs).
* **Live SQL:** Get Data → SQL Server / PostgreSQL → table `v_master` (view in `sql/schema.sql`) plus `batches`, `predictions`, `alerts`.
  Power Query M for SQLite via ODBC or for SQL Server:
  `Sql.Database("SERVER","ZeroC"){[Schema="dbo",Item="v_master"]}[Data]`
* Schedule refresh in Power BI Service (on-premise gateway keeps data inside ABT-CPD).

## 2. Model (star schema)
`dim_date[date]` 1→* `fact_complaints[complaint_date]` · `dim_product[product_name]` 1→* `fact_complaints[product]` and `fact_risk_forecast[product]` ·
`dim_customer[customer_name]` 1→* `fact_complaints[customer]` · `fact_batches[batch_id]` 1→* `fact_complaints[batch_id]`.

## 3. DAX measures
```DAX
Complaints = COUNTROWS(fact_complaints)
Units (M) = DIVIDE(SUM(fact_batches[units_produced]), 1000000)
Complaints per Million Units = DIVIDE([Complaints], [Units (M)])
Repeat Rate % = DIVIDE(SUM(fact_complaints[is_repeat]), [Complaints])
CAPA Effectiveness % = DIVIDE(CALCULATE(COUNTROWS(fact_complaints), fact_complaints[effective] = 1), CALCULATE(COUNTROWS(fact_complaints), NOT ISBLANK(fact_complaints[effective])))
Avg Closure Days = AVERAGE(fact_complaints[closure_days])
Complaints PY = CALCULATE([Complaints], SAMEPERIODLASTYEAR(dim_date[date]))
YoY % = DIVIDE([Complaints] - [Complaints PY], [Complaints PY])
Severity Weighted = SUMX(fact_complaints, fact_complaints[severity])
-- Pareto cumulative %
Cum % = VAR cur = [Complaints]
  VAR tot = CALCULATE([Complaints], ALL(fact_complaints[category]))
  RETURN DIVIDE(SUMX(FILTER(ALL(fact_complaints[category]), [Complaints] >= cur), [Complaints]), tot)
-- SPC limits (c-chart)
Mean/Month = AVERAGEX(VALUES(dim_date[ym]), [Complaints])
UCL = [Mean/Month] + 3 * SQRT([Mean/Month])
Max 30d Risk = MAX(fact_risk_forecast[p30])
High-Risk Pairs = CALCULATE(COUNTROWS(fact_risk_forecast), fact_risk_forecast[p30] >= 0.4)
```

## 4. Report pages
1. **Command Centre** – KPI cards, Pareto (category, `Cum %` line), line chart with `UCL`, alert table (`fact_alerts`).
2. **Patterns** – matrix product × category (heat-map conditional formatting), decomposition tree on root cause (6M), chronic table (`fact_chronic`).
3. **Risk & Prediction** – matrix of `p30/p60/p90` by product × category, product and customer risk scores with RAG icons.
4. **KPI Scorecard** – `fact_kpi` with target lines (Zero-Complaint road map).
Set a **Data alert** on the *High-Risk Pairs* card (Service → card → ⋯ → Manage alerts) to e-mail the QA head when it is above 0.
'''
