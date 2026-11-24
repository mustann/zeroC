-- Example analytical queries (run in the app's SQL console, or in SSMS / pgAdmin / Power BI "Native query")

-- 1. Pareto by category (last 36 months)
SELECT category, COUNT(*) AS n, ROUND(100.0*SUM(COUNT(*)) OVER (ORDER BY COUNT(*) DESC)/SUM(COUNT(*)) OVER (),1) AS cum_pct
FROM v_master WHERE complaint_date >= date('2026-11-24','-36 months') GROUP BY category ORDER BY n DESC;

-- 2. Recurrence: complaints that repeat the same product + defect within 12 months
SELECT product, category, complaint_date,
       COUNT(*) OVER (PARTITION BY product, category ORDER BY julianday(complaint_date)
                      RANGE BETWEEN 365 PRECEDING AND 1 PRECEDING) AS repeats_in_prev_12m
FROM v_master ORDER BY complaint_date DESC LIMIT 50;

-- 3. CAPA effectiveness by action
SELECT category, action, COUNT(*) AS n, ROUND(100.0*AVG(effective),1) AS pct_effective
FROM v_master WHERE effective IS NOT NULL GROUP BY category, action ORDER BY category, pct_effective DESC;

-- 4. Complaints per million units, by product and year
SELECT pr.product_name, u.yr, u.units, c.n AS complaints, ROUND(1e6*c.n/u.units,2) AS per_million
FROM (SELECT product_id, strftime('%Y',batch_date) AS yr, SUM(units_produced) AS units FROM batches GROUP BY 1,2) u
JOIN (SELECT product_id, strftime('%Y',complaint_date) AS yr, COUNT(*) AS n FROM complaints GROUP BY 1,2) c USING (product_id, yr)
JOIN products pr USING (product_id) ORDER BY 1,2;
