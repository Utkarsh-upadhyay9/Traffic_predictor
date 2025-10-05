# Traffic Predictor v4.1 - Quick Start
# Starts backend server and opens frontend

Write-Host "🚀 Starting Traffic Predictor v4.1..." -ForegroundColor Green
Write-Host ""

# Start backend in hidden window
Write-Host "Starting backend on http://localhost:8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoProfile", "-Command", "cd '$PSScriptRoot'; python run_backend.py" -WindowStyle Hidden

# Wait for backend to start
Write-Host "Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

# Open frontend
Write-Host "Opening frontend in browser..." -ForegroundColor Cyan
Start-Process "http://localhost:8000" -ErrorAction SilentlyContinue
Start-Process "$PSScriptRoot\index.html"

Write-Host ""
Write-Host "✅ Traffic Predictor is running!" -ForegroundColor Green
Write-Host "   Backend: http://localhost:8000" -ForegroundColor White
Write-Host "   Frontend: index.html (opened in browser)" -ForegroundColor White
Write-Host ""
Write-Host "📚 See QUICK_START_V4.md for usage guide" -ForegroundColor Cyan
Write-Host ""
