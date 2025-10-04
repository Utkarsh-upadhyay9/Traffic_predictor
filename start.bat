@echo off
REM SimCity AI - Quick Start Script

echo.
echo ================================================
echo    SimCity AI - Starting Application
echo ================================================
echo.

echo [1/3] Starting Backend...
start "Backend-API" /MIN cmd /k "cd /d %~dp0 && venv\Scripts\activate && set SKIP_AUTH_VERIFICATION=true && python backend\main.py"

echo [2/3] Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

echo [3/3] Starting Frontend...
echo.
echo ================================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:3000
echo   API Docs: http://localhost:8000/docs
echo ================================================
echo.
echo Browser will open automatically...
echo.

cd frontend
npm start
