#!/usr/bin/env bash
set -e
# ── ZeroC-AI Launcher ──────────────────────────────────
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
pip install -r requirements.txt
streamlit run app.py
