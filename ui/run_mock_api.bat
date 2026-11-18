@echo off
REM DEV-ONLY Mock API Server for testing ui/app.py locally before Laptop 1 is ready.
REM DO NOT use during final evaluation!

cd /d "%~dp0"

echo ========================================================
echo  PathSnap DEV MOCK API (Local UI Testing Only)
echo  Listening on http://0.0.0.0:8000
echo ========================================================
python -m uvicorn mock_api:app --host 0.0.0.0 --port 8000 --reload
