"""Central configuration: paths, taxonomy, thresholds."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "zeroc.db"
SQL_DIR, MODEL_DIR, PBI_DIR, OUTBOX = ROOT / "sql", ROOT / "models", ROOT / "powerbi", ROOT / "outbox"
TODAY = pd.Timestamp("2026-11-24")          # prototype "as-of" date; use pd.Timestamp.today() in production
START = pd.Timestamp("2008-01-01")           # ~19 years of history

PRODUCTS = [f"Product {x}" for x in "ABCDEF"]
CUSTOMERS = [f"Customer {x}" for x in "ABCDEFGH"]      # anonymised
LINES = ["L1", "L2", "L3", "L4"]
SEVERITY = {"Foreign matter": 5, "Contamination": 5, "Infestation": 4, "Physical defect": 3, "Process defect": 3,
            "Packaging": 2, "Sensory": 2, "Weight / quantity": 2, "Other": 1}
CATEGORIES = list(SEVERITY)
SUBS = {"Foreign matter": ["Metal fragment", "Plastic / glass / hair"], "Packaging": ["Seal failure", "Print / label error"],
        "Sensory": ["Taste / odour", "Texture / colour"], "Weight / quantity": ["Underweight", "Short count"],
        "Infestation": ["Live insects", "Webbing / mites"], "Contamination": ["Mould / microbial", "Chemical / allergen"],
        "Physical defect": ["Broken / crushed", "Shape / size"], "Process defect": ["Under / over baked", "Mix / ingredient"],
        "Other": ["General / service"]}
SIXM = ["Man", "Machine", "Material", "Method", "Measurement", "Environment"]

PROCESS_COLS = ["md_gap_days", "moisture_dev_sigma", "seal_temp_dev_sigma", "storage_days", "checkweigher_drift_sigma",
                "new_supplier_lot", "humidity_pct"]
PROCESS_LABELS = {"md_gap_days": "Days since metal-detector verification", "moisture_dev_sigma": "Moisture deviation (σ)",
                  "seal_temp_dev_sigma": "Seal-temperature deviation (σ)", "storage_days": "Raw-material storage (days)",
                  "checkweigher_drift_sigma": "Check-weigher drift (σ)", "new_supplier_lot": "New / unapproved supplier lot",
                  "humidity_pct": "Warehouse humidity (%)"}

P_HIGH, P_WATCH = 0.40, 0.25      # 30-day probability tiers
SCORE_RED, SCORE_AMBER = 75, 55   # composite risk score (0-100) tiers
LOW_CONF = 0.50                   # below this -> human review queue
CHRONIC_N = 4                     # >=3 repeats (= 4 complaints) in 12 months = chronic

SOP_RULES = {  # rule layer: SOP / HACCP checks attached to every recommendation
 "Foreign matter": "Verify HACCP CCP records: metal-detector test pieces, sieve and magnet logs for the last 3 shifts; hold implicated lots.",
 "Packaging": "Verify sealing jaw temperature / pressure and film spec vs pack specification; run seal-integrity test on retained samples.",
 "Sensory": "Re-check raw-material sensory release, shelf-life / FIFO and storage conditions; compare against retained samples.",
 "Weight / quantity": "Re-calibrate check-weigher with certified test weights; review filler set-points and reject log.",
 "Infestation": "Review pest-control log, warehouse hygiene and humidity; inspect incoming lots and FIFO; quarantine suspect stock.",
 "Contamination": "Re-validate changeover cleaning (ATP swab); check supplier COA, allergen controls and water / air ingress.",
 "Physical defect": "Inspect conveyor drop-heights, handling and cartoning; review packing cushion and stacking pattern.",
 "Process defect": "Check process profile against recipe, dosing accuracy and operator set-up sign-off.",
 "Other": "Review dispatch checklist, documentation and logistics partner performance."}
