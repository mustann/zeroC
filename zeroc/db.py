"""SQL layer. SQLite for the prototype; swap `connect()` for SQLAlchemy (PostgreSQL / MS SQL) in production."""
import sqlite3, pandas as pd
from .config import *

def connect(): 
    DB_PATH.parent.mkdir(exist_ok=True); return sqlite3.connect(DB_PATH)

def init_db(data):
    DB_PATH.unlink(missing_ok=True)
    with connect() as con:
        con.executescript((SQL_DIR / "schema.sql").read_text())
        pd.DataFrame(dict(product_id=[f"P{i+1}" for i in range(6)], product_name=PRODUCTS, family="Bakery / snacks")).to_sql("products", con, if_exists="append", index=False)
        pd.DataFrame(dict(customer_id=[f"C{i+1}" for i in range(8)], customer_name=CUSTOMERS, segment=["Modern trade"] * 4 + ["Distributor"] * 4)).to_sql("customers", con, if_exists="append", index=False)
        for t in ("batches", "complaints", "capa"):
            df = data[t].copy()
            for c in df.columns:
                if "date" in c: df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
            df.to_sql(t, con, if_exists="append", index=False)

def read(sql, params=()):
    with connect() as con: return pd.read_sql_query(sql, con, params=params)

def master():
    df = read("SELECT * FROM v_master"); df["complaint_date"] = pd.to_datetime(df["complaint_date"]); df["t"] = (df.complaint_date.dt.year - 2008) * 12 + df.complaint_date.dt.month - 1
    return df

def batches():
    df = read("SELECT b.*, p.product_name AS product FROM batches b JOIN products p USING(product_id)"); df["batch_date"] = pd.to_datetime(df.batch_date)
    df["t"] = (df.batch_date.dt.year - 2008) * 12 + df.batch_date.dt.month - 1; return df

def write(df, table, replace=True):
    with connect() as con: df.to_sql(table, con, if_exists="replace" if replace else "append", index=False) if replace else df.to_sql(table, con, if_exists="append", index=False)

def execute(sql, params=()):
    with connect() as con: con.execute(sql, params); con.commit()
