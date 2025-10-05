@echo off
echo ========================================
echo   DIGI SIM AI v3.1 - Quick Start
echo ========================================
echo.
echo NEW: Real route calculation with origin + destination
echo.

REM Stop any running processes
echo Stopping old processes...
taskkill /F /IM python.exe /T 2>nul
timeout /t 2 /nobreak >nul

REM Start backend
echo.
echo Starting backend server...
call venv\Scripts\activate.bat
set GEMINI_API_KEY=AIzaSyDQBIe6HG58TMe_HYdW6F9AShugLYWYbYk
set GOOGLE_MAPS_API_KEY=YOUR_KEY_HERE_OPTIONAL
set PYTHONPATH=backend

echo.
echo ========================================
echo   Backend Running on Port 8001
echo ========================================
echo.
echo How to use:
echo   1. Open index.html in your browser
echo   2. RIGHT-CLICK map = Set Origin (green)
echo   3. LEFT-CLICK map = Set Destination (blue)
echo   4. Select date and time
echo   5. Click "Calculate Route & Traffic"
echo.
echo NEW: See real distance and travel times!
echo.
echo Keep this window open!
echo ========================================
echo.

uvicorn backend.main:app --host 0.0.0.0 --port 8001
pause
