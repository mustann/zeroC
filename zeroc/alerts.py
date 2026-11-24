"""Alert mechanism: 30-day risk >= threshold or SPC signal -> alert record + e-mail (SMTP if configured, otherwise written to /outbox)."""
import os, smtplib, json, pandas as pd
from email.message import EmailMessage
from . import db, analytics as A, recommender
from .config import *

def notify(subject, body):
    to = os.getenv("ALERT_TO", "qa.head@example.com")
    if os.getenv("SMTP_HOST"):
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, os.getenv("SMTP_USER", "zeroc@localhost"), to
        msg.set_content(body)
        with smtplib.SMTP(os.getenv("SMTP_HOST"), int(os.getenv("SMTP_PORT", 587))) as s:
            s.starttls()
            s.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASS"))
            s.send_message(msg)
        return "sent via SMTP"
    OUTBOX.mkdir(exist_ok=True)
    f = OUTBOX / f"{pd.Timestamp.now():%Y%m%d_%H%M%S}_{abs(hash(subject)) % 9999}.txt"
    f.write_text(f"To: {to}\nSubject: {subject}\n\n{body}")
    return f"written to outbox ({f.name})"

def build(m, preds, threshold=P_WATCH):
    out = []
    for r in preds.sort_values("p30", ascending=False).itertuples():
        if r.p30 < threshold: continue
        lvl = "HIGH" if r.p30 >= P_HIGH else ("WATCH" if r.p30 >= P_WATCH else "MEDIUM")
        pl = recommender.plan(m, r.product, r.category)
        dr = ", ".join(f"{n}" for n, _ in json.loads(r.drivers)[:2]) if hasattr(r, "drivers") and r.drivers else "n/a"
        out.append((lvl, f"{r.product} | {r.category}", f"30-day complaint risk {r.p30:.0%} (60d {r.p60:.0%}, 90d {r.p90:.0%}). Drivers: {dr or 'n/a'}. Do now: {pl['action']} ({pl['evidence']}). SOP: {pl['sop']}"))
    s = A.spc(m)
    if s.signal.iloc[-1] or s.signal.iloc[-3:].any():
        out.append(("HIGH", "Plant total complaints", "SPC signal on monthly complaint count (UCL / run-rule / CUSUM). Investigate drift before more complaints arrive."))
    return out

def run(send=True, threshold=P_WATCH):
    m = A.add_recurrence(db.master())
    preds = db.read("SELECT * FROM predictions")
    today = str(TODAY.date())
    db.execute("DELETE FROM alerts WHERE status='New'")
    for lvl, ent, msg in build(m, preds, threshold=threshold):
        db.execute("INSERT INTO alerts(created,level,entity,message,status) VALUES(?,?,?,?,?)", (today, lvl, ent, msg, "New"))
        if send and lvl == "HIGH":
            notify(f"[ZeroC-AI] {lvl} risk: {ent}", msg)
