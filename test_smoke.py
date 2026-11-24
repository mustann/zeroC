"""ZeroC-AI · End-to-End Smoke Test.
Run: python test_smoke.py  OR  pytest test_smoke.py
"""
import pytest
import pandas as pd
from zeroc import db, nlp, analytics as A, model as ML, recommender as RC, alerts
from zeroc.config import DB_PATH, MODEL_DIR, CATEGORIES, PRODUCTS


def test_database():
    assert DB_PATH.exists(), f"Database not found at {DB_PATH}"
    m = db.master()
    assert len(m) > 500, f"Expected >500 complaints, got {len(m)}"
    b = db.batches()
    assert len(b) > 500, f"Expected >500 batches, got {len(b)}"
    v = db.read("SELECT * FROM v_master LIMIT 5")
    assert len(v) == 5


def test_nlp_classifier():
    clf = nlp.load()
    res, cp = nlp.predict(clf, ["metal wire piece found in packaging pouch"])
    assert len(res) == 1
    assert "category" in res.columns
    assert "subdefect" in res.columns
    assert res.iloc[0]["category"] in CATEGORIES
    words = nlp.explain(clf, "metal wire piece found in pack")
    assert isinstance(words, list)


def test_analytics():
    m = A.add_recurrence(db.master())
    b = db.batches()

    p = A.pareto(m, "category")
    assert len(p) > 0
    assert "cum_pct" in p.columns

    ch = A.chronic(m)
    assert len(ch) > 0

    rec = A.recurrence_detailed(m)
    assert len(rec) > 0
    assert "status" in rec.columns

    lift, ct = A.lift_matrix(m, "product")
    assert not lift.empty

    s = A.spc(m)
    assert len(s) > 0
    assert "signal" in s.columns

    kp = A.kpis(m, b)
    assert len(kp) > 0


def test_models_and_recommender():
    m = A.add_recurrence(db.master())
    b = db.batches()

    recs = RC.recommend(m, "Foreign matter", "Product B")
    assert len(recs) > 0
    assert "action" in recs.columns

    sim = RC.similar_cases(m, "wire piece", k=2)
    assert len(sim) == 2

    # Batch model scoring
    import joblib
    bm = joblib.load(MODEL_DIR / "batch_models.joblib")
    reading = ML.baseline_reading(b, "Product A")
    sc, anyp, dec, drv = ML.score_batch(bm, reading)
    assert 0.0 <= anyp <= 1.0
    assert dec in ["HOLD LOT - QA release required", "ENHANCED CHECKS before dispatch", "RELEASE - normal checks"]


def test_alerts():
    m = A.add_recurrence(db.master())
    preds = db.read("SELECT * FROM predictions")
    al = alerts.build(m, preds)
    assert isinstance(al, list)


if __name__ == "__main__":
    print("Running ZeroC-AI smoke tests...")
    test_database()
    print(" [x] Database check passed")
    test_nlp_classifier()
    print(" [x] NLP hierarchical classification passed")
    test_analytics()
    print(" [x] Analytics (Pareto, Chronic, SPC, KPIs, Lift) passed")
    test_models_and_recommender()
    print(" [x] Recommender and batch risk model passed")
    test_alerts()
    print(" [x] Alert engine check passed")
    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")
