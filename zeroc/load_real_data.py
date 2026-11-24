"""
ZeroC-AI · Real-Data Loader
============================
Optional module – connects to PostgreSQL *or* reads CSV files.

Set the env-vars below to activate:
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS   → PostgreSQL
    CSV_DIR                                        → folder of CSVs
"""

from __future__ import annotations
import os, pathlib
import pandas as pd


def load_real() -> dict[str, pd.DataFrame]:
    """Return the same dict[str, DataFrame] contract as the SQLite loader."""
    if os.getenv("CSV_DIR"):
        return _from_csv(pathlib.Path(os.environ["CSV_DIR"]))
    return _from_pg()


# ── CSV ──────────────────────────────────────────────────────────────

def _from_csv(folder: pathlib.Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("batches", "lab_tests", "complaints", "suppliers"):
        p = folder / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p)
        else:
            out[name] = pd.DataFrame()
    return out


# ── PostgreSQL ───────────────────────────────────────────────────────

def _from_pg() -> dict[str, pd.DataFrame]:
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError("pip install psycopg2-binary") from exc

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASS"],
    )
    out = {}
    for name in ("batches", "lab_tests", "complaints", "suppliers"):
        out[name] = pd.read_sql(f"SELECT * FROM {name}", conn)
    conn.close()
    return out
