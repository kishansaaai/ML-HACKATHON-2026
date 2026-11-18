@echo off
setlocal
pushd "%~dp0\.."
set "PYTHONPATH=%CD%\src"
if not exist ".venv\Scripts\python.exe" (
  echo Missing .venv. Follow the setup steps in README.md first.
  popd
  exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
set "EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %EXIT_CODE%
