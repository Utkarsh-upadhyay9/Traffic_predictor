# Start SimCity AI Backend with Calendar Integration
Write-Host "`n🚀 Starting SimCity AI v2.1 - Calendar-Integrated Traffic Predictor`n" -ForegroundColor Cyan
Write-Host "📅 Features:" -ForegroundColor Yellow
Write-Host "  ✅ Click-to-place pins on map" -ForegroundColor Green
Write-Host "  ✅ Google Calendar holiday integration" -ForegroundColor Green
Write-Host "  ✅ 30-day holiday sync" -ForegroundColor Green
Write-Host "  ✅ Location-based predictions" -ForegroundColor Green
Write-Host "`n"

# Stop any existing processes
Write-Host "🛑 Stopping old processes..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Activate virtual environment and start
Write-Host "🔧 Starting backend on port 8001...`n" -ForegroundColor Cyan
cd C:\Users\utkar\Desktop\Xapps\Digi_sim
.\venv\Scripts\Activate.ps1
$env:GEMINI_API_KEY='AIzaSyDQBIe6HG58TMe_HYdW6F9AShugLYWYbYk'

# Run uvicorn directly instead of through main.py
uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
