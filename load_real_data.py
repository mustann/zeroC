"""Load the organisers' REAL ABT-CPD complaint file into the ZeroC-AI database.

Usage:
    python load_real_data.py path/to/complaints.xlsx [batches_or_volumes.csv]

1. Edit COLUMN_MAP so the keys match YOUR Excel/CSV headers and the values are ZeroC-AI field names.
2. Required columns: date, customer, product, complaint_text.
   If category is provided, it is stored as category_manual (expert seed).
   If category is absent, the NLP classifier automatically classifies it.
3. Optional columns: batch_id, line, root_cause, root_cause_6m, action, owner, due_date, effective [1/0 or Y/N], closure_days.
4. Optional batch/process columns: units_produced, md_gap_days, moisture_dev_sigma, seal_temp_dev_sigma, storage_days, checkweigher_drift_sigma, humidity_pct.
5. After loading, the script automatically retrains the models and refreshes predictions and Power BI exports.
"""
from __future__ import annotations

import os, sys, pathlib
import numpy as np
import pandas as pd
from zeroc.config import DB_PATH, PRODUCTS, CUSTOMERS, PROCESS_COLS
from zeroc import db, pipeline, nlp

COLUMN_MAP = {
    # YOUR column header -> ZeroC-AI standard field
    "Complaint Date": "complaint_date",
    "Date": "complaint_date",
    "Customer Name": "customer",
    "Customer": "customer",
    "Product": "product",
    "Product Name": "product",
    "Complaint Description": "complaint_text",
    "Description": "complaint_text",
    "Text": "complaint_text",
    "Complaint Category": "category",
    "Category": "category",
    "Sub Defect": "subdefect",
    "Sub-defect": "subdefect",
    "Subdefect": "subdefect",
    "Batch No": "batch_id",
    "Batch": "batch_id",
    "Line": "line",
    "Root Cause": "root_cause",
    "6M": "root_cause_6m",
    "Root Cause 6M": "root_cause_6m",
    "Corrective Action": "action",
    "Action": "action",
    "Action Effective": "effective",
    "Effective": "effective",
    "Closure Days": "closure_days",
    "Owner": "owner",
    "Due Date": "due_date",
}

REQUIRED = ["complaint_date", "customer", "product", "complaint_text"]


def load_file(path: str | pathlib.Path, batch_path: str | pathlib.Path | None = None):
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    print(f"Reading {p}...")
    if p.suffix.lower() in [".xlsx", ".xls"]:
        raw = pd.read_excel(p)
    else:
        raw = pd.read_csv(p)

    df = raw.rename(columns=COLUMN_MAP)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. Available: {list(raw.columns)}")

    df["complaint_date"] = pd.to_datetime(df["complaint_date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("complaint_date").reset_index(drop=True)

    # Ensure complaint_id
    if "complaint_id" not in df.columns:
        df["complaint_id"] = [i + 1 for i in range(len(df))]

    # Products & Customers normalization
    prods = sorted(df["product"].dropna().unique())
    custs = sorted(df["customer"].dropna().unique())
    prod_map = {name: f"P{i+1}" for i, name in enumerate(prods)}
    cust_map = {name: f"C{i+1}" for i, name in enumerate(custs)}

    # Connect to DB and recreate schema if needed
    with db.connect() as conn:
        schema = (pathlib.Path(__file__).resolve().parent / "sql" / "schema.sql").read_text()
        conn.executescript(schema)
        # Populate products & customers
        pd.DataFrame([{"product_id": pid, "product_name": p, "family": "CPD"} for p, pid in prod_map.items()]).to_sql("products", conn, if_exists="replace", index=False)
        pd.DataFrame([{"customer_id": cid, "customer_name": c, "segment": "Trade"} for c, cid in cust_map.items()]).to_sql("customers", conn, if_exists="replace", index=False)

        # Prepare complaints table
        c_df = pd.DataFrame()
        c_df["complaint_id"] = df["complaint_id"]
        c_df["complaint_date"] = df["complaint_date"]
        c_df["customer_id"] = df["customer"].map(cust_map)
        c_df["product_id"] = df["product"].map(prod_map)
        c_df["batch_id"] = df.get("batch_id", None)
        c_df["complaint_text"] = df["complaint_text"]
        c_df["category_manual"] = df.get("category", None)
        c_df["subdefect_manual"] = df.get("subdefect", None)
        c_df["category_pred"] = df.get("category", None)
        c_df["subdefect_pred"] = df.get("subdefect", None)
        c_df["confidence"] = 1.0 if "category" in df.columns else None
        c_df["reviewed"] = 1 if "category" in df.columns else 0
        c_df.to_sql("complaints", conn, if_exists="replace", index=False)

        # Prepare CAPA table
        capa_df = pd.DataFrame()
        capa_df["capa_id"] = range(1, len(df) + 1)
        capa_df["complaint_id"] = df["complaint_id"]
        capa_df["root_cause_6m"] = df.get("root_cause_6m", "Method")
        capa_df["root_cause"] = df.get("root_cause", "Unspecified")
        capa_df["action"] = df.get("action", "Standard verification")
        capa_df["owner"] = df.get("owner", "QA Lead")
        capa_df["due_date"] = df.get("due_date", (pd.to_datetime(df["complaint_date"]) + pd.Timedelta(days=14)).dt.strftime("%Y-%m-%d"))
        capa_df["status"] = df.get("status", "Closed")

        if "effective" in df.columns:
            eff = df["effective"].map(lambda x: 1 if str(x).upper() in ["1", "Y", "YES", "TRUE"] else (0 if str(x).upper() in ["0", "N", "NO", "FALSE"] else None))
            capa_df["effective"] = eff
        else:
            capa_df["effective"] = 1

        capa_df["closure_days"] = df.get("closure_days", 7)
        capa_df.to_sql("capa", conn, if_exists="replace", index=False)

        # Batches table
        if batch_path and pathlib.Path(batch_path).exists():
            bp = pathlib.Path(batch_path)
            b_raw = pd.read_excel(bp) if bp.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(bp)
            b_df = b_raw.rename(columns=COLUMN_MAP)
            if "product_id" not in b_df.columns and "product" in b_df.columns:
                b_df["product_id"] = b_df["product"].map(prod_map)
            b_df.to_sql("batches", conn, if_exists="replace", index=False)
        else:
            # Create synthetic batch context around the complaints dates
            cur_batches = pd.read_sql("SELECT COUNT(*) FROM batches", conn).iloc[0, 0]
            if cur_batches == 0:
                print("Generating linked batch baseline for production volumes...")
                from zeroc.generate_data import _batches
                # generate standard batches
                b_df = _batches(3000)
                b_df.to_sql("batches", conn, if_exists="replace", index=False)

    print(f"Loaded {len(df)} complaints into {DB_PATH.name}.")
    print("Retraining models and refreshing pipeline...")
    os.environ["ZC_DATA"] = "real"
    pipeline.run_nlp()
    pipeline.run_models()
    pipeline.export_powerbi.export()
    print("All models and Power BI files updated successfully!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    load_file(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
