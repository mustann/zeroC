"""Synthetic 19-year dataset (batches, process data, free-text complaints, CAPA) used until real ABT-CPD data is loaded.
Process parameters genuinely influence complaint probability so the ML models have real signal to learn."""
import numpy as np, pandas as pd
from .config import *

rng = np.random.default_rng(2026)
BATCHES_PM = 6                                     # batches per product per month
BASE = dict(zip(CATEGORIES, [1.5, 1.2, 1.4, 1.0, 0.8, 0.5, 0.9, 0.8, 0.4]))   # relative frequency of each category
PW = [1.6, 1.1, 1.0, .8, .8, .7]

TEMPL = {
 ("Foreign matter", "Metal fragment"): ["metal piece found in {p} pack", "small steel wire inside {p}", "staple like metallic fragment noticed while eating {p}", "screw particle in {p} carton"],
 ("Foreign matter", "Plastic / glass / hair"): ["hair strand seen in {p}", "piece of plastic found inside {p} pouch", "glass particle reported in {p}", "thread and paper fibre inside {p}"],
 ("Packaging", "Seal failure"): ["{p} pouch seal open and product leaking", "seal burst on {p} packs at store", "pack torn at seal, {p} spilled"],
 ("Packaging", "Print / label error"): ["wrong date coding on {p} carton", "batch label smudged on {p}", "barcode print missing on {p} outer"],
 ("Sensory", "Taste / odour"): ["{p} has stale taste and off odour", "rancid smell from {p} pack", "bitter flavour in {p}, customer returned"],
 ("Sensory", "Texture / colour"): ["{p} texture soft and soggy", "{p} colour too dark and burnt look", "hard texture, {p} not crisp"],
 ("Weight / quantity", "Underweight"): ["{p} packs underweight versus declared grams", "net weight less than label on {p}", "customer weighed {p}, short in grams"],
 ("Weight / quantity", "Short count"): ["short count of pieces in {p} case", "pieces missing in {p} pack", "less packs than invoiced in {p} carton"],
 ("Infestation", "Live insects"): ["live insects found in {p}", "weevil infestation seen in {p} stock", "beetle and larvae inside {p} pack"],
 ("Infestation", "Webbing / mites"): ["webbing and mites visible in {p}", "infested {p} with web like strands", "mite infestation reported in {p}"],
 ("Contamination", "Mould / microbial"): ["mould growth visible on {p}", "fungal spots found in {p} pack", "spoilage and yeast smell in {p}"],
 ("Contamination", "Chemical / allergen"): ["oil and chemical smell, contaminated {p}", "cleaning detergent odour in {p}", "possible allergen cross contact in {p}"],
 ("Physical defect", "Broken / crushed"): ["{p} broken and crushed on receipt", "cracked pieces, crumbled {p} in carton", "damaged and chipped {p} at warehouse"],
 ("Physical defect", "Shape / size"): ["{p} deformed shape, uneven size", "misshaped {p}, thickness not uniform", "size variation across {p} pieces"],
 ("Process defect", "Under / over baked"): ["{p} found raw and underbaked", "overbaked {p} with burnt edge", "doughy undercooked centre in {p}"],
 ("Process defect", "Mix / ingredient"): ["salt dosage error in {p}", "sugar mix wrong in {p} batch", "ingredient missing in {p} formulation"],
 ("Other", "General / service"): ["delay in supply of {p} and invoice mismatch", "documentation missing with {p} consignment", "courier service issue for {p}"]}
PRE = ["", "", "Customer reported: ", "Retailer feedback - ", "Consumer call: ", "Urgent: "]
SUF = ["", "", " - photo attached", " - sample returned", ". Please investigate.", " (see batch note)"]
VAGUE = ["customer unhappy with {p}, quality not as expected", "{p} not acceptable, please check", "complaint about {p} received from store", "problem noticed in {p}, details to follow"]
NOISE = ["pack otherwise ok", "second time this month", "store manager complained", "no injury reported"]

CAUSE = {"Foreign matter": [("Machine", "Sieve / magnet integrity breach"), ("Method", "Metal-detector verification skipped"), ("Material", "Foreign matter in incoming raw material")],
 "Packaging": [("Machine", "Sealing jaw temperature drift"), ("Method", "Wrong film / pack spec used"), ("Material", "Packing film thickness variation")],
 "Sensory": [("Material", "Raw-material quality variation"), ("Method", "Storage / ageing beyond limit"), ("Environment", "High humidity in packing hall")],
 "Weight / quantity": [("Measurement", "Check-weigher calibration drift"), ("Machine", "Feeder / filler variation")],
 "Infestation": [("Environment", "Warehouse hygiene and humidity"), ("Material", "Infested raw-material lot"), ("Method", "FIFO not followed")],
 "Contamination": [("Method", "Cleaning / changeover validation gap"), ("Environment", "Airborne / water ingress"), ("Material", "Supplier lot contamination")],
 "Physical defect": [("Machine", "Conveyor / drop-height damage"), ("Man", "Careless manual handling")],
 "Process defect": [("Method", "Process profile out of range"), ("Man", "Operator dosing error")],
 "Other": [("Man", "Order / dispatch error")]}
ACT = {"Foreign matter": [("Sieve + magnet + metal-detector verification each shift", .86), ("Supplier audit and incoming sieving", .74), ("Operator retraining only", .30), ("Increase visual inspection", .35)],
 "Packaging": [("Sealing-jaw PM + temperature alarm", .84), ("Film spec lock and incoming film test", .72), ("Pack-by-pack operator checks", .33), ("Retraining only", .28)],
 "Sensory": [("Raw-material sensory release test", .80), ("Shelf-life / FIFO control", .70), ("Supplier warning letter", .40), ("Retraining only", .25)],
 "Weight / quantity": [("Check-weigher recalibration + auto-reject", .88), ("Filler servo tuning", .70), ("Manual weight audits", .38)],
 "Infestation": [("Pest-control programme + warehouse deep clean", .85), ("Incoming lot fumigation check", .76), ("Storage limit 30 days + FIFO audit", .70), ("Spray only", .32)],
 "Contamination": [("Changeover cleaning validation (ATP swab)", .84), ("Supplier lot COA verification", .72), ("Retraining only", .30)],
 "Physical defect": [("Drop-height / conveyor redesign", .80), ("Handling SOP + retraining", .55), ("Extra packing cushion", .40)],
 "Process defect": [("Process-profile SPC with alarms", .85), ("Dosing poka-yoke", .78), ("Retraining only", .30)],
 "Other": [("Dispatch checklist", .60), ("Courier review", .50)]}
OWNERS = ["QA Head", "Plant Manager", "Maintenance Lead", "Packing Supervisor", "Stores In-charge"]

def _batches():
    months = pd.date_range(START, TODAY, freq="MS"); T = len(months); rows = []; n = 0
    for ti, m in enumerate(months):
        left = T - 1 - ti                                  # months before today (0 = current month)
        season = np.sin((m.month - 2) * np.pi / 6)
        for p in range(6):
            for _ in range(BATCHES_PM):
                n += 1; r = dict(batch_id=f"B{m:%y%m}{n % 10000:04d}", product_id=f"P{p+1}", line=LINES[(p + (rng.random() < .3)) % 4],
                    batch_date=m + pd.Timedelta(days=int(rng.integers(0, 27))), units_produced=int(rng.integers(180_000, 260_000)))
                md = rng.exponential(3.0); mo = rng.normal(0, 1); se = rng.normal(0, 1); st = rng.gamma(4, 5); cw = rng.normal(0, 1)
                if left < 7 and p == 1: md += rng.exponential(5)            # Product B: metal-detector verification gaps
                if left < 4 and p == 5: mo += 1.3                            # Product F: moisture drift
                if left < 8 and p == 3: se += 1.1                            # Product D: seal temperature drift
                if left < 5 and p == 2: st += 25                             # Product C: long storage
                if left < 10 and p == 0: cw += 1.2                           # Product A: check-weigher drift
                r.update(md_gap_days=round(md, 1), moisture_dev_sigma=round(mo, 2), seal_temp_dev_sigma=round(se, 2), storage_days=round(st, 1),
                         checkweigher_drift_sigma=round(cw, 2), new_supplier_lot=int(rng.random() < .06),
                         humidity_pct=round(60 + 8 * season + rng.normal(0, 5), 1), moisture_pct=round(3.5 + .3 * mo + rng.normal(0, .1), 2),
                         water_activity=round(.32 + .02 * mo + rng.normal(0, .01), 3), net_weight_avg_g=round(100 - .8 * cw + rng.normal(0, .5), 2))
                rows.append(r)
    return pd.DataFrame(rows)

def _p(r, c, p, t, T):
    z = np.log(.045 * BASE[c] * PW[p]) - .55 * t / T
    z += {"Infestation": .6, "Sensory": .4, "Contamination": .4}.get(c, 0) * np.sin((r.batch_date.month - 2) * np.pi / 6)
    z += {"Foreign matter": .45 * max(0, r.md_gap_days - 6) + .8 * r.new_supplier_lot, "Sensory": .55 * max(0, abs(r.moisture_dev_sigma) - .8) + .015 * max(0, r.storage_days - 30),
          "Packaging": 1.1 * max(0, abs(r.seal_temp_dev_sigma) - .6), "Weight / quantity": 1.0 * max(0, abs(r.checkweigher_drift_sigma) - 1),
          "Infestation": .06 * max(0, r.storage_days - 30) + .03 * max(0, r.humidity_pct - 65), "Contamination": .8 * r.new_supplier_lot + .02 * max(0, r.humidity_pct - 68)}.get(c, 0)
    return 1 / (1 + np.exp(-z))

def generate():
    b = _batches(); T = b.batch_date.dt.to_period("M").nunique(); b["t"] = (b.batch_date.dt.year - 2008) * 12 + b.batch_date.dt.month - 1
    comp, capa = [], []
    for r in b.itertuples():
        p = int(r.product_id[1:]) - 1
        for c in CATEGORIES:
            if rng.random() >= _p(r, c, p, r.t, T): continue
            d = r.batch_date + pd.Timedelta(days=int(rng.integers(10, 70)))
            if d > TODAY: continue                                         # complaint not yet received (right-censoring)
            sub = str(rng.choice(SUBS[c])); txt = str(rng.choice(TEMPL[(c, sub)])).format(p=f"Product {'ABCDEF'[p]}")
            u = rng.random()
            if u < .10: txt = str(rng.choice(VAGUE)).format(p=f"Product {'ABCDEF'[p]}")              # vague / ambiguous wording
            elif u < .22:                                                                              # mixed complaint (secondary issue)
                ks = [k for k in TEMPL if k[0] != c]; o = ks[int(rng.integers(len(ks)))]; txt += " and also " + str(rng.choice(TEMPL[o])).format(p="it")
            if rng.random() < .12:                                                                     # typo
                w = txt.split(); j = int(rng.integers(len(w))); w[j] = w[j][:-1] if len(w[j]) > 3 else w[j]; txt = " ".join(w)
            txt = rng.choice(PRE) + txt + rng.choice(SUF) + (". " + str(rng.choice(NOISE)) if rng.random() < .15 else "")
            cu = (p * 3 + CATEGORIES.index(c)) % 8 if rng.random() < .45 else int(rng.integers(0, 8))
            comp.append(dict(complaint_date=d, customer_id=f"C{cu+1}", product_id=r.product_id, batch_id=r.batch_id, complaint_text=str(txt), _cat=c, _sub=sub, _r=r))
    comp.sort(key=lambda x: x["complaint_date"]); rows = []
    for i, x in enumerate(comp, 1):
        r, c = x.pop("_r"), x["_cat"]; causes = CAUSE[c]; k = int(rng.integers(len(causes)))
        if c == "Foreign matter" and r.md_gap_days > 6 and rng.random() < .7: k = 1
        if c == "Weight / quantity" and abs(r.checkweigher_drift_sigma) > 1 and rng.random() < .7: k = 0
        if c == "Packaging" and abs(r.seal_temp_dev_sigma) > .8 and rng.random() < .7: k = 0
        if c == "Infestation" and r.storage_days > 30 and rng.random() < .6: k = 2
        a, pr = ACT[c][int(rng.integers(len(ACT[c])))]; age = (TODAY - x["complaint_date"]).days
        capa.append(dict(capa_id=i, complaint_id=i, root_cause_6m=causes[k][0], root_cause=causes[k][1], action=a, owner=str(rng.choice(OWNERS)),
            due_date=x["complaint_date"] + pd.Timedelta(days=30), status="Closed" if age > 45 else ("In progress" if age > 15 else "Open"),
            effective=(int(rng.random() < pr) if age > 120 else None), closure_days=int(7 + rng.random() * (30 if c == "Foreign matter" else 22))))
        lab = rng.random() < .30                                           # only ~30% carry an expert label (seed set)
        rows.append(dict(complaint_id=i, **{k: v for k, v in x.items() if not k.startswith("_")}, category_manual=c if lab else None,
                         subdefect_manual=x["_sub"] if lab else None, _true_cat=c, _true_sub=x["_sub"]))
    cdf = pd.DataFrame(rows); truth = cdf[["complaint_id", "_true_cat", "_true_sub"]]
    return dict(batches=b.drop(columns="t"), complaints=cdf.drop(columns=["_true_cat", "_true_sub"]), capa=pd.DataFrame(capa), truth=truth)
