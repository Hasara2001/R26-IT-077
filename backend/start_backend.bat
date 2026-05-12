@echo off
echo ============================================
echo  Liver Image Feature Extraction - Backend
echo ============================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10 from python.org
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies (first time only)...
pip install -r requirements.txt

echo.
echo [3/3] Starting backend server on http://localhost:8000
echo       Press CTRL+C to stop
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
