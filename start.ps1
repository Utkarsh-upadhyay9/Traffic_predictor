# ============================================
# SimCity AI - Simple One-Command Startup
# ============================================
# Starts both backend and frontend together

Write-Host "`n"
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - Starting Application" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "`n"

# Activate virtual environment
Write-Host " [1/3] Activating Python environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Set environment variables
$env:SKIP_AUTH_VERIFICATION = "true"

# Start backend in background
Write-Host " [2/3] Starting Backend (FastAPI)..." -ForegroundColor Yellow
$backend = Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; `$env:SKIP_AUTH_VERIFICATION='true'; Write-Host 'Backend Server' -ForegroundColor Green; python backend/main.py" -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 5

# Start frontend
Write-Host " [3/3] Starting Frontend (React)..." -ForegroundColor Yellow
Write-Host "`n"
Write-Host "================================================" -ForegroundColor Green
Write-Host "         SERVERS STARTING!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Green
Write-Host "`n"
Write-Host " Backend:  http://localhost:8000" -ForegroundColor Cyan
Write-Host " Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host " API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "`n"
Write-Host " Opening browser in 5 seconds..." -ForegroundColor Yellow
Write-Host "`n"

# Start frontend in current terminal (will show live updates)
cd frontend
npm start
