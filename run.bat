@echo off
REM ── ZeroC-AI Launcher ──────────────────────────────────
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
pip install -r requirements.txt
streamlit run app.py
pause
