-- ZeroC-AI relational schema (SQLite syntax; runs unchanged on PostgreSQL / MS SQL with minor type edits)
CREATE TABLE IF NOT EXISTS products   (product_id TEXT PRIMARY KEY, product_name TEXT, family TEXT);
CREATE TABLE IF NOT EXISTS customers  (customer_id TEXT PRIMARY KEY, customer_name TEXT, segment TEXT);
CREATE TABLE IF NOT EXISTS batches (
  batch_id TEXT PRIMARY KEY, product_id TEXT REFERENCES products(product_id), line TEXT, batch_date DATE, units_produced INTEGER,
  md_gap_days REAL, moisture_dev_sigma REAL, seal_temp_dev_sigma REAL, storage_days REAL, checkweigher_drift_sigma REAL,
  new_supplier_lot INTEGER, humidity_pct REAL, moisture_pct REAL, water_activity REAL, net_weight_avg_g REAL);
CREATE TABLE IF NOT EXISTS complaints (
  complaint_id INTEGER PRIMARY KEY, complaint_date DATE, customer_id TEXT REFERENCES customers(customer_id),
  product_id TEXT REFERENCES products(product_id), batch_id TEXT, complaint_text TEXT,
  category_manual TEXT, subdefect_manual TEXT,            -- expert-labelled seed (about 30% of records)
  category_pred TEXT, subdefect_pred TEXT, confidence REAL, reviewed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS capa (
  capa_id INTEGER PRIMARY KEY, complaint_id INTEGER REFERENCES complaints(complaint_id), root_cause_6m TEXT, root_cause TEXT,
  action TEXT, owner TEXT, due_date DATE, status TEXT, effective INTEGER, closure_days INTEGER);   -- effective: 1/0, NULL = pending 90-day check
CREATE TABLE IF NOT EXISTS predictions (run_at TEXT, product TEXT, category TEXT, p30 REAL, p60 REAL, p90 REAL, drivers TEXT);
CREATE TABLE IF NOT EXISTS backtest   (horizon INTEGER, t_month TEXT, product TEXT, category TEXT, p REAL, y INTEGER);
CREATE TABLE IF NOT EXISTS themes     (cluster INTEGER, size INTEGER, top_terms TEXT, dominant_category TEXT, purity REAL, avg_conf REAL, new_theme INTEGER);
CREATE TABLE IF NOT EXISTS alerts     (alert_id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, level TEXT, entity TEXT, message TEXT, status TEXT);

-- Master view used by Python analytics AND Power BI (final category = expert label if present, else NLP prediction)
CREATE VIEW IF NOT EXISTS v_master AS
SELECT c.complaint_id, c.complaint_date, strftime('%Y-%m', c.complaint_date) AS ym, CAST(strftime('%Y', c.complaint_date) AS INT) AS yr,
       p.product_name AS product, cu.customer_name AS customer, c.batch_id, b.line, c.complaint_text,
       COALESCE(c.category_manual, c.category_pred) AS category, COALESCE(c.subdefect_manual, c.subdefect_pred) AS subdefect,
       c.confidence, c.reviewed, k.root_cause_6m, k.root_cause, k.action, k.owner, k.due_date, k.status, k.effective, k.closure_days
FROM complaints c JOIN products p ON p.product_id = c.product_id JOIN customers cu ON cu.customer_id = c.customer_id
LEFT JOIN batches b ON b.batch_id = c.batch_id LEFT JOIN capa k ON k.complaint_id = c.complaint_id;
