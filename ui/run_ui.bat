@echo off
REM PathSnap Evaluator UI Launcher (Laptop 2 - Windows)
REM Usage:
REM   run_ui.bat
REM   run_ui.bat http://192.168.1.25:8000

cd /d "%~dp0"

if not "%~1"=="" (
    set "PATHSNAP_API_URL=%~1"
)

if "%PATHSNAP_API_URL%"=="" (
    set "PATHSNAP_API_URL=http://127.0.0.1:8000"
)

echo ========================================================
echo  PathSnap Evaluator UI (Laptop 2)
echo  Target API: %PATHSNAP_API_URL%
echo ========================================================
python -m streamlit run app.py --server.port 8501 --server.headless false
