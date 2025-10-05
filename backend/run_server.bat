@echo off
cd /d "%~dp0"
echo Starting Traffic Predictor Backend...
python -m uvicorn simple_backend:app --host 0.0.0.0 --port 8000
pause
