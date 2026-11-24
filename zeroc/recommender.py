"""Action recommender: rank corrective / preventive actions by PROVEN effectiveness in similar past cases (Wilson lower bound),
retrieve the most similar historical complaints (TF-IDF cosine) and attach the SOP / HACCP rule layer."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import *
from .nlp import clean

def wilson(k, n, z=1.64):
    if n == 0: return 0.0
    p = k / n; return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)

def recommend(m, category, product=None):
    d = m[(m.category == category) & m.effective.notna() & m.action.notna()]
    g = d.groupby("action").effective.agg(["sum", "count"]).rename(columns={"sum": "worked", "count": "cases"})
    if product: g2 = d[d["product"] == product].groupby("action").effective.agg(["sum", "count"]).rename(columns={"sum": "worked_here", "count": "cases_here"}); g = g.join(g2).fillna(0)
    g["success_rate"] = g.worked / g.cases; g["confidence_adj"] = [wilson(k, n) for k, n in zip(g.worked, g.cases)]
    g["verdict"] = np.where(g.confidence_adj >= .6, "RECOMMENDED", np.where(g.success_rate < .45, "AVOID alone", "SUPPORTING")); return g.sort_values("confidence_adj", ascending=False).reset_index()

def similar_cases(m, text, k=3):
    d = m[m.effective.notna()].reset_index(drop=True); v = TfidfVectorizer(ngram_range=(1, 2)).fit(d.complaint_text.map(clean)); s = cosine_similarity(v.transform([clean(text)]), v.transform(d.complaint_text.map(clean)))[0]
    i = np.argsort(-s)[:k]; return d.iloc[i][["complaint_date", "product", "complaint_text", "root_cause", "action", "effective"]].assign(similarity=s[i].round(2))

def failed_actions(m, min_cases=10):
    g = m[m.effective.notna()].groupby(["category", "action"]).effective.agg(["mean", "count"]).reset_index(); return g[(g["mean"] < .5) & (g["count"] >= min_cases)].sort_values("mean")

def plan(m, product, category):
    r = recommend(m, category, product); top = r.iloc[0] if len(r) else None
    return dict(action=None if top is None else top.action, evidence=None if top is None else f"worked in {int(top.worked)} of {int(top.cases)} similar cases", sop=SOP_RULES[category])
