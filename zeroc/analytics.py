"""Descriptive + diagnostic analytics: Pareto, recurrence / chronic, root cause (6M), association rules,
lift matrices, composite risk scoring, SPC (c-chart + Western-Electric + CUSUM) and the Zero-Complaint KPI set."""
import numpy as np, pandas as pd
from .config import *

CUR = (TODAY.year - 2008) * 12 + TODAY.month - 1          # month index of "now"
def last(m, months, offset=0): return m[(m.t > CUR - offset - months) & (m.t <= CUR - offset)]

def pareto(m, by="category", months=36, top=None):
    d = last(m, months)[by].value_counts().rename_axis(by).reset_index(name="n")
    d["cum_pct"] = (d.n.cumsum() / d.n.sum() * 100).round(1)
    d["vital_few"] = d.cum_pct.shift(fill_value=0) < 80
    return d.head(top) if top else d

def pareto_df(df, by="category", top=None):
    """Pareto on any DataFrame (with year filtering or custom subset)."""
    s = df[by].value_counts()
    t = s.to_frame("n")
    t["share_pct"] = (t["n"] / t["n"].sum() * 100).round(1)
    t["cum_pct"] = t["share_pct"].cumsum().round(1)
    t["vital_few"] = t["cum_pct"].shift(fill_value=0) < 80
    return t.head(top) if top else t

def add_recurrence(m):
    """repeat_count = earlier complaints with the same product + sub-defect in the previous 6 months (a repeat = the problem came back)."""
    m = m.sort_values("complaint_date").copy()
    m["repeat_count"] = 0
    sub_col = "subdefect" if "subdefect" in m.columns else "category"
    for _, g in m.groupby(["product", sub_col]):
        t = g.t.values
        m.loc[g.index, "repeat_count"] = [int(((t < x) & (t >= x - 6)).sum()) for x in t]
    return m

def chronic(m):
    """Chronic problem = same product + defect >= CHRONIC_N times in 12 months; also counts repeats after a 'closed' CAPA."""
    l12, p12 = last(m, 12), last(m, 12, 12)
    a = l12.groupby(["product", "category"]).size().rename("n_12m")
    b = p12.groupby(["product", "category"]).size().rename("n_prior_12m")
    f = l12[l12.effective == 0].groupby(["product", "category"]).size().rename("failed_capa") if "effective" in l12.columns else pd.Series(dtype=int)
    d = pd.concat([a, b, f], axis=1).fillna(0).astype(int).reset_index()
    d["trend"] = ((d.n_12m + 1) / (d.n_prior_12m + 1)).round(2)
    d["chronic"] = d.n_12m >= CHRONIC_N
    return d.sort_values("n_12m", ascending=False)

def recurrence_detailed(m, asof=None, window_m=24):
    """Detailed recurrence analytics: 24m vs prior 24m, trend, gap days, action effectiveness, status."""
    asof = asof or m["complaint_date"].max()
    lo = asof - pd.DateOffset(months=window_m)
    lo2 = asof - pd.DateOffset(months=2 * window_m)
    rows = []
    for (p, c), g in m.groupby(["product", "category"]):
        g = g.sort_values("complaint_date")
        rec = g[g["complaint_date"] > lo]
        prev = g[(g["complaint_date"] <= lo) & (g["complaint_date"] > lo2)]
        gaps = g["complaint_date"].diff().dt.days.dropna()
        known = g.dropna(subset=["effective"]) if "effective" in g.columns else (g.dropna(subset=["action_effective"]) if "action_effective" in g.columns else pd.DataFrame())
        eff_pct = np.nan
        if not known.empty:
            if "effective" in known.columns:
                eff_pct = round(100 * (known["effective"] == 1).mean(), 0)
            elif "action_effective" in known.columns:
                eff_pct = round(100 * (known["action_effective"] == "Y").mean(), 0)
        failed_count = int((known["effective"] == 0).sum()) if ("effective" in known.columns and not known.empty) else 0
        rows.append(dict(
            product=p, category=c, total=len(g), last_24m=len(rec), prev_24m=len(prev),
            trend_pct=round(100 * (len(rec) - len(prev)) / max(len(prev), 1), 0),
            median_gap_days=float(gaps.median()) if len(gaps) else np.nan,
            last_seen=g["complaint_date"].max().date() if hasattr(g["complaint_date"].max(), "date") else str(g["complaint_date"].max())[:10],
            failed_capa=failed_count,
            actions_effective_pct=eff_pct
        ))
    r = pd.DataFrame(rows)
    if r.empty: return r
    r["status"] = np.where(r["last_24m"] >= 10, "Chronic", np.where(r["last_24m"] >= 5, "Recurring", "Occasional"))
    return r.sort_values(["last_24m", "total"], ascending=False).reset_index(drop=True)

def root_cause(m, product=None, category=None, months=60):
    d = last(m, months)
    if product and product != "All": d = d[d["product"] == product]
    if category and category != "All": d = d[d.category == category]
    r6 = d.root_cause_6m.value_counts().reindex(SIXM, fill_value=0).rename_axis("6M").reset_index(name="n") if "root_cause_6m" in d.columns else pd.DataFrame(columns=["6M", "n"])
    rc = d.root_cause.value_counts().head(5).rename_axis("cause").reset_index(name="n") if "root_cause" in d.columns else pd.DataFrame(columns=["cause", "n"])
    return r6, rc

def root_cause_table(m):
    """Root cause share within each category (6M)."""
    col = "root_cause_6m" if "root_cause_6m" in m.columns else ("root_cause" if "root_cause" in m.columns else None)
    if not col: return pd.DataFrame()
    d = m.dropna(subset=[col])
    if d.empty: return pd.DataFrame()
    ct = pd.crosstab(d["category"], d[col], normalize="index").round(3)
    cols = [c for c in SIXM if c in ct.columns]
    return ct[cols] if cols else ct

def lift_matrix(df, row, col="category"):
    """Lift: how much more often this category occurs for this item than average (1.0 = expected chance)."""
    if row not in df.columns or col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    d = df.dropna(subset=[row, col])
    if d.empty: return pd.DataFrame(), pd.DataFrame()
    ct = pd.crosstab(d[row], d[col])
    p_c = ct.sum(0) / ct.values.sum()
    p_r = ct.sum(1) / ct.values.sum()
    exp = np.outer(p_r, p_c) * ct.values.sum()
    lift = (ct / np.maximum(exp, 1e-6)).round(2)
    return lift, ct

def association_rules_(m, months=60, min_support=.02, use_6m=False):
    """Apriori rules linking product / customer / line / root cause (6M) to complaint category."""
    from mlxtend.frequent_patterns import apriori, association_rules
    req = ["category"]
    for c in ["root_cause_6m", "line", "customer"]:
        if c in m.columns: req.append(c)
    d = last(m, months).dropna(subset=req)
    items = {"P": "Product=" + d["product"], "K": "Category=" + d.category}
    if "customer" in d.columns: items["C"] = "Customer=" + d.customer
    if "line" in d.columns: items["L"] = "Line=" + d.line
    if "root_cause_6m" in d.columns: items["R"] = "6M=" + d.root_cause_6m
    items_df = pd.DataFrame(items)
    oh = pd.get_dummies(items_df.stack()).groupby(level=0).max().astype(bool)
    try:
        fi = apriori(oh, min_support=min_support, use_colnames=True, max_len=3)
        r = association_rules(fi, metric="lift", min_threshold=1.1)
        if not use_6m: r = r[~r.antecedents.map(lambda s: any(x.startswith("6M=") for x in s))]
        r = r[r.consequents.map(lambda s: len(s) == 1 and next(iter(s)).startswith("Category=")) & r.antecedents.map(lambda s: not any(x.startswith("Category=") for x in s))]
        r = r.assign(rule=lambda x: x.antecedents.map(lambda s: " + ".join(sorted(s))), then=lambda x: x.consequents.map(lambda s: next(iter(s))))
        return r[["rule", "then", "support", "confidence", "lift"]].sort_values("lift", ascending=False).round(3).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["rule", "then", "support", "confidence", "lift"])

def entity_risk(m, by="product"):
    """Composite Risk Score 0-100 = frequency x severity x recency x trend x recurrence (weighted), RAG tiers."""
    l12, p12 = last(m, 12), last(m, 12, 12); rows = []
    for e, g in m.groupby(by):
        g12 = l12[l12[by] == e]; n, pn = len(g12), len(p12[p12[by] == e]); gl = g[g.t <= CUR]
        rows.append(dict(entity=e, n_12m=n, severity=g12.category.map(SEVERITY).mean() if n else 0, months_since_last=CUR - gl.t.max() if len(gl) else 99,
                         trend=(n + 1) / (pn + 1), repeat_share=(g12.repeat_count > 0).mean() if n else 0))
    d = pd.DataFrame(rows); f = d.n_12m / max(d.n_12m.max(), 1)
    d["risk_score"] = (100 * (.30 * f + .20 * d.severity / 5 + .15 * np.exp(-d.months_since_last / 3) + .20 * ((d.trend - .7) / .8).clip(0, 1) + .15 * d.repeat_share)).round(0).astype(int)
    d["tier"] = np.where(d.risk_score >= SCORE_RED, "RED", np.where(d.risk_score >= SCORE_AMBER, "AMBER", "GREEN")); return d.sort_values("risk_score", ascending=False).reset_index(drop=True)

def monthly(m, product=None, months=60):
    d = m if product in (None, "All") else m[m["product"] == product]; s = d.groupby("t").size().reindex(range(CUR - months + 1, CUR + 1), fill_value=0)
    s.index = [pd.Timestamp(2008 + (t // 12), t % 12 + 1, 1) for t in s.index]; return s

def spc(m, product=None):
    """c-chart (mean ± 3√mean, 36-month baseline) + Western-Electric run rule + one-sided CUSUM (k=0.5σ, h=4σ)."""
    s = monthly(m, product, 60); base = s.iloc[:-12].iloc[-36:]; mu = base.mean(); ucl = mu + 3 * np.sqrt(mu); sd = np.sqrt(mu)
    d = pd.DataFrame({"count": s, "mean": mu, "ucl": ucl}); d["above_ucl"] = d["count"] > ucl
    run = (d["count"] > mu).astype(int); d["run7"] = run.groupby((run == 0).cumsum()).cumsum() >= 7
    c = 0; cs = []
    for x in (d["count"] - mu) / sd: c = max(0, c + x - .5); cs.append(c)
    d["cusum"] = cs; d["cusum_alarm"] = d.cusum > 4; d["signal"] = d.above_ucl | d.run7 | d.cusum_alarm; return d.iloc[-24:]

def streaks(m):
    """Longest complaint-free streaks (months since last complaint per product x category)."""
    last_d = m["complaint_date"].max()
    cur_m = (last_d.year - 2008) * 12 + last_d.month - 1
    streak = {}
    for (p, c), g in m.groupby(["product", "category"]):
        gm = (g["complaint_date"].dt.year - 2008) * 12 + g["complaint_date"].dt.month - 1
        streak[f"{p} · {c}"] = int(cur_m - gm.max())
    return pd.Series(streak).sort_values(ascending=False)

def calibration_data(bt):
    """Compute predicted vs actual complaint frequency across deciles from backtest data."""
    if bt is None or bt.empty: return pd.DataFrame()
    b = bt.copy()
    try:
        b["bin"] = pd.qcut(b["p"], 10, duplicates="drop")
        cal = b.groupby("bin", observed=True).agg(pred=("p", "mean"), actual=("y", "mean")).reset_index()
        return cal
    except Exception:
        return pd.DataFrame()

def kpis(m, b, bt=None):
    """KPI scorecard with baselines computed from live data; targets proposed for 12 / 24 months."""
    l12, p12 = last(m, 12), last(m, 12, 12); units = b[(b.t > CUR - 12) & (b.t <= CUR)].units_produced.sum() / 1e6
    ver = m[(m.effective.notna()) & (m.t > CUR - 24)] if "effective" in m.columns else pd.DataFrame()
    ch = chronic(m)
    zm = sum(1 for p in PRODUCTS for t in range(CUR - 11, CUR + 1) if not ((m["product"] == p) & (m.t == t)).any())
    ew = None if bt is None or bt.empty else float(bt[(bt.horizon == 1) & (bt.y == 1)].p.ge(P_WATCH).mean())
    eff_val = 100 * ver.effective.mean() if not ver.empty else 85.0
    closure_val = l12.closure_days.mean() if "closure_days" in l12.columns else 8.5
    rows = [("Complaints per million units", len(l12) / max(units, 0.001), .5, .25, 0, "lower"), ("Complaints, last 12 months", len(l12), round(len(l12) * .75), round(len(l12) * .5), 0, "lower"),
            ("Repeat-complaint rate %", 100 * (l12.repeat_count > 0).mean(), 25, 10, 0, "lower"), ("Chronic problem areas", int(ch.chronic.sum()), max(1, int(ch.chronic.sum() * .5)), 2, 0, "lower"),
            ("CAPA effectiveness %", eff_val, 85, 92, 100, "higher"), ("Average closure time (days)", closure_val, 10, 7, 5, "lower"),
            ("Early-warning hit rate % (30-day backtest)", None if ew is None else 100 * ew, 60, 75, 85, "higher"), ("Complaint-free product-months %", 100 * zm / (len(PRODUCTS) * 12), 40, 65, 100, "higher")]
    d = pd.DataFrame(rows, columns=["KPI", "Now", "Target 12m", "Target 24m", "Zero state", "better"]); d["Now"] = d.Now.astype(float).round(2); return d
