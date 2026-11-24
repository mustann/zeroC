"""Predictive analytics (XGBoost) + explainability (SHAP).
 A) Pair model  : P(complaint of category c for product p in the next 30 / 60 / 90 days) from lags, trend, season and process drift.
 B) Batch model : P(complaint of category c | this batch's process & lab readings) for pre-dispatch release decisions.
 Validation is strictly time-based (train past -> calibrate -> test latest 12 months). Probabilities are isotonic-calibrated."""
import json, joblib, numpy as np, pandas as pd, xgboost as xgb, shap
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from .config import *
from .analytics import CUR

NICE = {"n1": "Complaints last month", "n3": "Complaints last 3 months", "n6": "Complaints last 6 months", "n12": "Complaints last 12 months", "prev12": "Complaints prior 12 months",
        "trend": "Year-on-year trend", "since_last": "Months since last complaint", "moy_s": "Season (sin)", "moy_c": "Season (cos)", "sev": "Severity class", "fail12": "Failed CAPAs (12 m)", "proc_exp1": "Process readings: expected complaints (this month)", "proc_exp3": "Process readings: expected complaints (3 months)",
        "cat_id": "Complaint category", "prod_id": "Product", **{f"d_{k}": "Recent drift: " + v for k, v in PROCESS_LABELS.items()}, **{k: v for k, v in PROCESS_LABELS.items()}}
XGB = dict(n_estimators=220, max_depth=3, learning_rate=.05, subsample=.85, colsample_bytree=.85, min_child_weight=3, reg_lambda=2, eval_metric="logloss", random_state=1, n_jobs=2)
MONO = {"proc_exp1": 1, "proc_exp3": 1, "n3": 1, "n12": 1, "trend": 1}
PAIR_FEATS = ["proc_exp1", "proc_exp3", "n3", "n12", "trend", "since_last", "fail12", "sev", "moy_s", "moy_c"] + ["d_" + c for c in ["md_gap_days", "moisture_dev_sigma", "seal_temp_dev_sigma", "storage_days", "checkweigher_drift_sigma", "new_supplier_lot", "humidity_pct"]]
XGBP = dict(n_estimators=120, max_depth=2, learning_rate=.05, subsample=.8, colsample_bytree=.8, min_child_weight=10, reg_lambda=5, random_state=1, n_jobs=2,
            monotone_constraints="(" + ",".join(str(MONO.get(f, 0)) for f in PAIR_FEATS) + ")")
def _logit(p): p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p)).reshape(-1, 1)
class Platt:
    """Smooth probability calibration (Platt scaling) fitted on the validation window."""
    def __init__(self, raw, y): self.lr = LogisticRegression(C=10).fit(_logit(raw), y)
    def predict(self, raw): return self.lr.predict_proba(_logit(raw))[:, 1]
def _drift(b):   # monthly process drift per product (3-month rolling)
    d = b.assign(md=b.md_gap_days, mo=b.moisture_dev_sigma.abs(), se=b.seal_temp_dev_sigma.abs(), st=b.storage_days, cw=b.checkweigher_drift_sigma.abs(), ns=b.new_supplier_lot, hu=b.humidity_pct)
    g = d.groupby(["product", "t"])[["md", "mo", "se", "st", "cw", "ns", "hu"]].mean(); out = {}
    for p in PRODUCTS: out[p] = g.loc[p].reindex(range(0, CUR + 1)).ffill().rolling(3, min_periods=1).mean()
    return out
DR = ["md_gap_days", "moisture_dev_sigma", "seal_temp_dev_sigma", "storage_days", "checkweigher_drift_sigma", "new_supplier_lot", "humidity_pct"]

def batch_oof(m, b):
    """Out-of-fold batch-model probabilities (time blocks) -> expected complaints per product/category/month from process readings."""
    X = _bx(b); X = X[X.t <= CUR].copy(); lab = m.dropna(subset=["batch_id"]); E = np.zeros((6, 9, CUR + 1)); blk = X.t // 48
    for ci, c in enumerate(CATEGORIES):
        y = X.batch_id.isin(lab[lab.category == c].batch_id).astype(int).values; p = np.zeros(len(X))
        for k in blk.unique():
            tr = (blk != k).values & (X.t <= CUR - 2).values
            if y[tr].sum() < 20: p[(blk == k).values] = y[tr].mean(); continue
            p[(blk == k).values] = xgb.XGBClassifier(**{**XGB, "n_estimators": 90}).fit(X.loc[tr, BF], y[tr]).predict_proba(X.loc[(blk == k).values, BF])[:, 1]
        np.add.at(E[:, ci, :], (X.prod_id.values, X.t.values), p)
    return E

def pair_features(m, b):
    E = batch_oof(m, b); T = CUR + 1; arr = np.zeros((6, 9, T)); fail = np.zeros((6, 9, T))
    for r in m[(m.t <= CUR)].itertuples():
        arr[PRODUCTS.index(r.product), CATEGORIES.index(r.category), r.t] += 1; fail[PRODUCTS.index(r.product), CATEGORIES.index(r.category), r.t] += (r.effective == 0)
    cs = np.concatenate([np.zeros((6, 9, 1)), arr.cumsum(2)], 2); cf = np.concatenate([np.zeros((6, 9, 1)), fail.cumsum(2)], 2); dr = _drift(b); rows = []
    for p in range(6):
        for c in range(9):
            lastc = -99
            for t in range(24, T):
                if arr[p, c, t - 1] > 0: lastc = t - 1
                w = lambda k: cs[p, c, t + 1] - cs[p, c, t + 1 - k]
                r = dict(prod_id=p, cat_id=c, t=t, n1=arr[p, c, t], n3=w(3), n6=w(6), n12=w(12), prev12=cs[p, c, t + 1 - 12] - cs[p, c, t + 1 - 24], sev=SEVERITY[CATEGORIES[c]],
                         fail12=cf[p, c, t + 1] - cf[p, c, t + 1 - 12], since_last=min(60, t - lastc), moy_s=np.sin(2 * np.pi * (t % 12) / 12), moy_c=np.cos(2 * np.pi * (t % 12) / 12))
                r["trend"] = (r["n12"] + 1) / (r["prev12"] + 1); r["proc_exp1"] = E[p, c, t]; r["proc_exp3"] = E[p, c, t - 2: t + 1].sum()
                for k, col in zip(["md", "mo", "se", "st", "cw", "ns", "hu"], DR): r["d_" + col] = dr[PRODUCTS[p]].loc[t, k]
                rows.append(r)
    F = pd.DataFrame(rows); F["product"] = F.prod_id.map(dict(enumerate(PRODUCTS))); F["category"] = F.cat_id.map(dict(enumerate(CATEGORIES)))
    for h in (1, 2, 3):
        fut = np.array([arr[r.prod_id, r.cat_id, r.t + 1: r.t + 1 + h].sum() if r.t + h <= CUR else np.nan for r in F.itertuples()]); F[f"y{h}"] = np.where(np.isnan(fut), np.nan, (fut > 0).astype(float))
    return F

FEATS = None
def train_pair_models(m, b):
    global FEATS; F = pair_features(m, b); FEATS = PAIR_FEATS
    models, metrics, bt, E = {}, {}, [], CUR
    for h in (1, 2, 3):
        y = f"y{h}"; D = F.dropna(subset=[y]); te = D[D.t > E - h - 12]; va = D[(D.t > E - h - 24) & (D.t <= E - h - 12)]; tr = D[D.t <= E - h - 24]
        mdl = xgb.XGBClassifier(**XGBP).fit(tr[FEATS], tr[y]); iso = Platt(mdl.predict_proba(va[FEATS])[:, 1], va[y])
        p = iso.predict(mdl.predict_proba(te[FEATS])[:, 1]); yt = te[y].values; k = max(1, int(len(te) * .10)); top = np.argsort(-p)[:k]
        metrics[f"{h*30}d"] = dict(pr_auc=round(average_precision_score(yt, p), 3), baseline_pr_auc=round(average_precision_score(yt, te.n12 + .1 * te.n3), 3), roc_auc=round(roc_auc_score(yt, p), 3),
                                   recall_top10pct=round(float(yt[top].sum() / max(yt.sum(), 1)), 3), brier=round(brier_score_loss(yt, p), 4), positive_rate=round(float(yt.mean()), 3), n_test=len(te))
        # final refit on everything that has a label (time-ordered), keep validation-based calibrator
        full = xgb.XGBClassifier(**XGBP).fit(D[FEATS], D[y]); models[h] = (full, iso)
        if h == 1: bt = pd.DataFrame(dict(horizon=h, t_month=te.t.map(lambda t: f"{2008 + t // 12}-{t % 12 + 1:02d}"), product=te["product"], category=te.category, p=p, y=yt.astype(int)))
    joblib.dump(models, MODEL_DIR / "pair_models.joblib"); return models, metrics, bt, F

def predict_now(models, F):
    now = F[F.t == CUR].copy(); out = now[["product", "category"]].reset_index(drop=True)
    for h, col in zip((1, 2, 3), ("p30", "p60", "p90")): mdl, iso = models[h]; out[col] = iso.predict(mdl.predict_proba(now[FEATS])[:, 1])
    ex = shap.TreeExplainer(models[1][0]); sv = ex.shap_values(now[FEATS]); dr = []
    for i in range(len(now)):
        o = [j for j in np.argsort(-sv[i]) if FEATS[j] not in ("sev",)][:3]; dr.append(json.dumps([(NICE.get(FEATS[j], FEATS[j]), round(float(sv[i][j]), 3)) for j in o if sv[i][j] > 0.02]))
    out["drivers"] = dr; out.insert(0, "run_at", datetime.now().isoformat(timespec="seconds")); return out

def shap_for(models, F, product, category):
    now = F[(F.t == CUR) & (F["product"] == product) & (F.category == category)]; sv = shap.TreeExplainer(models[1][0]).shap_values(now[FEATS])[0]
    return pd.DataFrame({"driver": [NICE.get(f, f) for f in FEATS], "value": now[FEATS].iloc[0].values, "shap": sv}).query("driver not in ['Complaint category', 'Product']").sort_values("shap", key=abs, ascending=False).head(8)

# ---------------- batch model ----------------
BF = PROCESS_COLS + ["prod_id", "moy_s", "moy_c"]
def _bx(b):
    X = b.copy(); X["prod_id"] = X["product"].map({p: i for i, p in enumerate(PRODUCTS)}); mo = X.batch_date.dt.month; X["moy_s"] = np.sin(2 * np.pi * mo / 12); X["moy_c"] = np.cos(2 * np.pi * mo / 12); return X

def train_batch_models(m, b):
    X = _bx(b); cut = CUR - 2; X = X[X.t <= cut]; lab = m.dropna(subset=["batch_id"]); models, mets = {}, {}
    for c in CATEGORIES:
        X[c] = X.batch_id.isin(lab[lab.category == c].batch_id).astype(int)
    tr, te = X[X.t <= cut - 24], X[X.t > cut - 24]
    for c in CATEGORIES:
        if tr[c].sum() < 25: continue
        mdl = xgb.XGBClassifier(**{**XGB, "n_estimators": 160}).fit(tr[BF], tr[c]); p = mdl.predict_proba(te[BF])[:, 1]
        mets[c] = dict(pr_auc=round(average_precision_score(te[c], p), 3), roc_auc=round(roc_auc_score(te[c], p), 3), base_rate=round(float(te[c].mean()), 4)) if te[c].sum() else {}
        models[c] = xgb.XGBClassifier(**{**XGB, "n_estimators": 160}).fit(X[BF], X[c])
    joblib.dump(models, MODEL_DIR / "batch_models.joblib"); return models, mets

def score_batch(models, reading):
    """reading: dict(product, month, + PROCESS_COLS). Returns per-category probabilities, release decision, and SHAP drivers per category."""
    r = pd.DataFrame([reading]); r["prod_id"] = PRODUCTS.index(reading["product"]); r["moy_s"] = np.sin(2 * np.pi * reading["month"] / 12); r["moy_c"] = np.cos(2 * np.pi * reading["month"] / 12)
    res, dr = {}, {}
    for c, mdl in models.items():
        res[c] = float(mdl.predict_proba(r[BF])[0, 1]); sv = shap.TreeExplainer(mdl).shap_values(r[BF])[0]
        dr[c] = sorted([(NICE.get(f, f), float(s)) for f, s in zip(BF, sv) if f in PROCESS_COLS], key=lambda x: -x[1])[:3]
    any_p = 1 - np.prod([1 - p * (.4 + .6 * SEVERITY[c] / 5) for c, p in res.items()]); s = pd.Series(res).sort_values(ascending=False)
    dec = "HOLD LOT - QA release required" if any_p >= .30 else "ENHANCED CHECKS before dispatch" if any_p >= .18 else "RELEASE - normal checks"
    return s, any_p, dec, dr

def baseline_reading(b, product): r = b[b["product"] == product].tail(60)[PROCESS_COLS].median().to_dict(); r.update(product=product, month=TODAY.month); return r
