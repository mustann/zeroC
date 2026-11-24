# ZeroC-AI · Power BI build guide
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
