# Start SimCity AI Full Stack Application
# Run this script to launch both backend and frontend

Write-Host "=" -ForegroundColor Cyan
Write-Host "🏙️  SimCity AI - Full Stack Startup" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-Not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "   Run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Check if ML models exist
if (-Not (Test-Path "ml/models")) {
    Write-Host "⚠️  ML models not found. Training now..." -ForegroundColor Yellow
    python ml/traffic_model.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Model training failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "✓ ML models found" -ForegroundColor Green
Write-Host ""

# Start backend
Write-Host "🚀 Starting Backend API..." -ForegroundColor Cyan
$env:SKIP_AUTH_VERIFICATION = "true"

Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    `$host.ui.RawUI.WindowTitle = 'SimCity AI - Backend';
    Write-Host '🔧 Backend Server' -ForegroundColor Green;
    Write-Host 'Port: 8000' -ForegroundColor Yellow;
    Write-Host 'Docs: http://localhost:8000/docs' -ForegroundColor Cyan;
    Write-Host '';
    cd '$PWD\backend';
    ..\venv\Scripts\Activate.ps1;
    `$env:SKIP_AUTH_VERIFICATION='true';
    python main.py
"

Write-Host "✓ Backend starting on http://localhost:8000" -ForegroundColor Green
Write-Host "   Swagger UI: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

# Wait for backend to start
Write-Host "⏳ Waiting for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if node_modules exists
if (-Not (Test-Path "frontend/node_modules")) {
    Write-Host "⚠️  Frontend dependencies not installed. Installing..." -ForegroundColor Yellow
    cd frontend
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ npm install failed!" -ForegroundColor Red
        exit 1
    }
    cd ..
}

# Start frontend
Write-Host "🚀 Starting Frontend..." -ForegroundColor Cyan

Start-Process powershell -ArgumentList "-NoExit", "-Command", "
    `$host.ui.RawUI.WindowTitle = 'SimCity AI - Frontend';
    Write-Host '🎨 Frontend Server' -ForegroundColor Green;
    Write-Host 'Port: 3000' -ForegroundColor Yellow;
    Write-Host 'URL: http://localhost:3000' -ForegroundColor Cyan;
    Write-Host '';
    cd '$PWD\frontend';
    npm start
"

Write-Host "✓ Frontend starting on http://localhost:3000" -ForegroundColor Green
Write-Host ""

Write-Host "=" -ForegroundColor Green
Write-Host "✅ SimCity AI is running!" -ForegroundColor Green
Write-Host "=" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Open your browser to: http://localhost:3000" -ForegroundColor Cyan
Write-Host "📚 API Documentation: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Yellow
Write-Host "   - Two terminal windows will open (Backend & Frontend)" -ForegroundColor White
Write-Host "   - Keep both running to use the app" -ForegroundColor White
Write-Host "   - Press Ctrl+C in each window to stop servers" -ForegroundColor White
Write-Host ""
Write-Host "🎉 Happy Simulating!" -ForegroundColor Magenta
