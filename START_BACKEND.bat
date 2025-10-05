@echo off
cd /d "%~dp0"
echo.
echo ========================================
echo   Digi_sim v3.0 Backend Server
echo ========================================
echo.
echo Starting backend on port 8001...
echo Keep this window OPEN!
echo.
echo Press Ctrl+C to stop the server
echo.

call venv\Scripts\activate.bat
set GEMINI_API_KEY=AIzaSyDQBIe6HG58TMe_HYdW6F9AShugLYWYbYk
set PYTHONPATH=backend

uvicorn backend.main:app --host 0.0.0.0 --port 8001

pause
