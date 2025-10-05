@echo off
echo Starting Traffic Predictor Backend...
echo.
cd /d "%~dp0backend"
python main.py
pause
