"""NLP: text cleaning, hierarchical complaint classifier (TF-IDF + Logistic Regression; optional Sentence-BERT),
active-learning review queue and unsupervised theme clustering (KMeans) to discover unlabelled complaint types."""
import re, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import make_pipeline
from .config import *

STOP = {"customer", "reported", "retailer", "feedback", "consumer", "call", "urgent", "photo", "attached", "sample", "returned", "please", "investigate", "see", "batch", "note", "pack", "otherwise", "ok", "second", "time", "this", "month", "store", "manager", "complained", "no", "injury", "reported"}
def clean(t): return " ".join(w for w in re.findall(r"[a-z]+", str(t).lower()) if w not in STOP and not re.fullmatch(r"product|[a-f]", w))

def _vec():
    try:                                                    # optional upgrade: semantic embeddings
        import os; assert os.getenv("ZEROC_SBERT") == "1"
        from sentence_transformers import SentenceTransformer
        from sklearn.base import BaseEstimator, TransformerMixin
        class SB(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None): return self
            def transform(self, X): return SentenceTransformer("all-MiniLM-L6-v2").encode(list(X))
        return SB()
    except Exception:
        return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)

def train(master):
    """Train on the expert-labelled seed. Label = 'Category|Sub-defect' so one model gives both levels."""
    lab = master[master.category_manual.notna()] if "category_manual" in master else master
    X = lab.complaint_text.map(clean); y = lab.category_manual + "|" + lab.subdefect_manual
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=7)
    mk = lambda: make_pipeline(_vec(), LogisticRegression(C=8, max_iter=2000, class_weight="balanced"))
    m = mk().fit(Xtr, ytr); pr = m.predict(Xte)
    cat = lambda s: pd.Series(s).str.split("|").str[0]
    metrics = dict(n_train=len(Xtr), n_test=len(Xte), macro_f1_category=round(f1_score(cat(yte), cat(pr), average="macro"), 3),
                   macro_f1_subdefect=round(f1_score(yte, pr, average="macro"), 3), accuracy=round(float((cat(yte).values == cat(pr).values).mean()), 3),
                   report=classification_report(cat(yte), cat(pr), output_dict=True, zero_division=0))
    m = mk().fit(X, y); MODEL_DIR.mkdir(exist_ok=True); joblib.dump(m, MODEL_DIR / "classifier.joblib")
    return m, metrics

def load(): return joblib.load(MODEL_DIR / "classifier.joblib")

def predict(model, texts):
    """Returns category, sub-defect, confidence (category prob = sum over its sub-defect classes) and full category probabilities."""
    P = model.predict_proba([clean(t) for t in texts]); cls = np.array(model.classes_); cats = np.array([c.split("|")[0] for c in cls])
    cp = pd.DataFrame({c: P[:, cats == c].sum(1) for c in CATEGORIES if (cats == c).any()})
    best = cp.idxmax(axis=1); sub = [cls[i].split("|")[1] for i in [np.argmax(np.where(cats == b, P[k], -1)) for k, b in enumerate(best)]]
    return pd.DataFrame({"category": best.values, "subdefect": sub, "confidence": cp.max(axis=1).values}), cp

def explain(model, text, top=5):
    """Top TF-IDF terms pushing the prediction (transparent, no black box)."""
    vec, lr = model[0], model[-1]
    if not hasattr(vec, "get_feature_names_out"): return []
    x = vec.transform([clean(text)]); i = int(np.argmax(lr.predict_proba(x)[0])); contrib = x.toarray()[0] * lr.coef_[i]
    names = vec.get_feature_names_out(); idx = np.argsort(-contrib)[:top]
    return [(names[j], float(contrib[j])) for j in idx if contrib[j] > 0]

def classify_all(model, con_df):
    out, _ = predict(model, con_df.complaint_text.tolist()); out.index = con_df.index; return out

def review_queue(master, n=15):
    """Active learning: lowest-confidence unlabelled complaints are sent to an expert; their labels retrain the model."""
    q = master[(master.category_manual.isna()) & (master.reviewed == 0)].sort_values("confidence"); return q.head(n)

def themes(master, k=10):
    """Cluster all complaint texts to expose recurring themes and spot clusters that do not map cleanly onto the taxonomy."""
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit(master.complaint_text.map(clean)); M = X.transform(master.complaint_text.map(clean))
    km = KMeans(k, n_init=10, random_state=3).fit(M); lab = km.labels_; terms = X.get_feature_names_out(); rows = []
    for c in range(k):
        s = master[lab == c]; top = [terms[i] for i in np.argsort(-km.cluster_centers_[c])[:5]]; dom = s.category.mode().iloc[0]; pur = float((s.category == dom).mean())
        rows.append(dict(cluster=c, size=len(s), top_terms=", ".join(top), dominant_category=dom, purity=round(pur, 2), avg_conf=round(float(s.confidence.fillna(1).mean()), 2), new_theme=int(pur < .75)))
    return pd.DataFrame(rows).sort_values("size", ascending=False), lab, M
