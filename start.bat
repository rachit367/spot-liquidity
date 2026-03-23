@echo off
title ICT Trading Bot
color 0A

echo.
echo  ================================================
echo   ICT Trading Bot  ^|  Dashboard + API
echo  ================================================
echo.

:: ── Check Python ─────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Install Python 3.11+ from https://python.org and add to PATH.
    echo.
    pause
    exit /b 1
)

:: ── Create venv if missing ────────────────────────
if not exist "venv\Scripts\activate.bat" (
    echo  [SETUP] Creating virtual environment...
    python -m venv venv
    echo  [SETUP] Done.
    echo.
)

:: ── Activate venv ─────────────────────────────────
call venv\Scripts\activate.bat
echo  [VENV]  Virtual environment activated.
echo.

:: ── Install / upgrade requirements ───────────────
echo  [SETUP] Installing dependencies ^(first run may take a minute^)...
pip install -r requirements.txt -q --disable-pip-version-check
echo  [SETUP] Dependencies ready.
echo.

:: ── Copy .env if missing ──────────────────────────
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo  [SETUP] Created .env from .env.example
    echo  [WARN]  Fill in your API keys in .env before going live.
    echo.
)

:: ── Launch dashboard ──────────────────────────────
echo  [START] Dashboard starting at http://localhost:8303
echo  [INFO]  Press Ctrl+C to stop.
echo.
start "" "http://localhost:8303"
python run_server.py --port 8303

pause
