"""End-to-end orchestration:  python -m zeroc.pipeline --init   (build everything)   |   --retrain   |   --ingest new.csv"""
import argparse, json, pandas as pd
from . import db, nlp, analytics as A, model as ML, recommender, alerts, export_powerbi
from .config import *
from .generate_data import generate

def run_nlp():
    m = db.master(); lab = m[m.complaint_id.isin(db.read("SELECT complaint_id FROM complaints WHERE category_manual IS NOT NULL").complaint_id)]
    seed = lab.assign(category_manual=lab.category, subdefect_manual=lab.subdefect); mdl, met = nlp.train(seed)
    all_ = db.read("SELECT complaint_id, complaint_text FROM complaints"); out = nlp.classify_all(mdl, all_); out["complaint_id"] = all_.complaint_id.values
    with db.connect() as con:
        con.executemany("UPDATE complaints SET category_pred=?, subdefect_pred=?, confidence=? WHERE complaint_id=?", [(r.category, r.subdefect, float(r.confidence), int(r.complaint_id)) for r in out.itertuples()])
    m = db.master(); th, _, _ = nlp.themes(m); db.write(th, "themes"); (MODEL_DIR / "nlp_metrics.json").write_text(json.dumps({k: v for k, v in met.items() if k != "report"}, indent=1)); return met

def run_models():
    m = A.add_recurrence(db.master()); b = db.batches(); pm, met, bt, F = ML.train_pair_models(m, b); pred = ML.predict_now(pm, F)
    db.write(pred, "predictions"); db.write(bt, "backtest"); bm, bmet = ML.train_batch_models(m, b)
    reg = MODEL_DIR / "registry.json"; hist = json.loads(reg.read_text()) if reg.exists() else []
    hist.append(dict(trained_at=pd.Timestamp.now().isoformat(timespec="seconds"), n_complaints=len(m), pair_model=met, batch_model=bmet)); reg.write_text(json.dumps(hist[-20:], indent=1))
    try:                                              # optional MLflow tracking
        import mlflow
        with mlflow.start_run(run_name="zeroc-retrain"): [mlflow.log_metric(f"pr_auc_{k}", v["pr_auc"]) for k, v in met.items()]
    except Exception: pass
    return met, bmet

def init(): 
    print("1/6 generating synthetic data..."); db.init_db(generate()); print("2/6 NLP classification + clustering..."); n = run_nlp(); print("   macro-F1:", n["macro_f1_category"])
    print("3/6 ML models (XGBoost + SHAP)..."); met, _ = run_models(); print("   30d PR-AUC:", met["30d"]["pr_auc"]); print("4/6 alerts..."); alerts.run(); print("5/6 Power BI export..."); export_powerbi.export(); print("6/6 done.")

def ingest(df, retrain=False):
    """Append new complaints (columns: date, product, customer, text[, batch]) -> auto-classify -> optionally retrain. Returns classified rows."""
    mdl = nlp.load(); res, _ = nlp.predict(mdl, df["text"].tolist()); pid = {p: f"P{i+1}" for i, p in enumerate(PRODUCTS)}; cid = {c: f"C{i+1}" for i, c in enumerate(CUSTOMERS)}
    nxt = int(db.read("SELECT COALESCE(MAX(complaint_id),0)+1 AS n FROM complaints").n[0])
    with db.connect() as con:
        for i, r in enumerate(df.itertuples()):
            con.execute("INSERT INTO complaints(complaint_id,complaint_date,customer_id,product_id,batch_id,complaint_text,category_pred,subdefect_pred,confidence,reviewed) VALUES(?,?,?,?,?,?,?,?,?,0)",
                        (nxt + i, str(pd.Timestamp(r.date).date()), cid.get(r.customer, "C1"), pid.get(r.product, "P1"), getattr(r, "batch", None), r.text, res.category[i], res.subdefect[i], float(res.confidence[i])))
            con.execute("INSERT INTO capa(complaint_id,status) VALUES(?, 'Open')", (nxt + i,))
    if retrain: run_models(); alerts.run()
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--init", action="store_true"); ap.add_argument("--retrain", action="store_true"); ap.add_argument("--ingest"); a = ap.parse_args()
    if a.init: init()
    if a.ingest: d = pd.read_csv(a.ingest); print(ingest(d, retrain=True))
    if a.retrain: run_nlp(); run_models(); alerts.run(); export_powerbi.export()
