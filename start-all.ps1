# ============================================
# SimCity AI - One-Terminal Startup Script
# ============================================
# Starts both backend and frontend in one terminal

Write-Host "`n" -NoNewline
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - Starting Full Application" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "`n"

# Get the script directory
$ROOT_DIR = $PSScriptRoot
if (-not $ROOT_DIR) {
    $ROOT_DIR = Get-Location
}

Write-Host " Workspace: $ROOT_DIR" -ForegroundColor Yellow
Write-Host "`n"

# Check if virtual environment exists
if (-not (Test-Path "$ROOT_DIR\venv\Scripts\Activate.ps1")) {
    Write-Host " ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host " Please run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Check if node_modules exists
if (-not (Test-Path "$ROOT_DIR\frontend\node_modules")) {
    Write-Host " WARNING: Frontend dependencies not installed!" -ForegroundColor Yellow
    Write-Host " Installing now..." -ForegroundColor Cyan
    Set-Location "$ROOT_DIR\frontend"
    npm install
    Set-Location $ROOT_DIR
}

Write-Host " Starting Backend Server..." -ForegroundColor Cyan
Write-Host " (FastAPI + ML Models + Gemini AI)" -ForegroundColor Gray
Write-Host "`n"

# Start backend in background job
$backendJob = Start-Job -ScriptBlock {
    param($rootDir)
    Set-Location $rootDir
    & "$rootDir\venv\Scripts\Activate.ps1"
    $env:SKIP_AUTH_VERIFICATION = "true"
    python backend/main.py
} -ArgumentList $ROOT_DIR

# Wait a bit for backend to start
Start-Sleep -Seconds 3

# Check if backend started
$backendOutput = Receive-Job -Job $backendJob
if ($backendOutput -match "error|Error|ERROR") {
    Write-Host " ERROR: Backend failed to start!" -ForegroundColor Red
    Write-Host $backendOutput
    Stop-Job -Job $backendJob
    Remove-Job -Job $backendJob
    exit 1
}

Write-Host " Backend: Starting on http://localhost:8000" -ForegroundColor Green
Write-Host "`n"

Write-Host " Starting Frontend Server..." -ForegroundColor Cyan
Write-Host " (React + Mapbox + UI)" -ForegroundColor Gray
Write-Host "`n"

# Start frontend in background job
$frontendJob = Start-Job -ScriptBlock {
    param($rootDir)
    Set-Location "$rootDir\frontend"
    $env:BROWSER = "none"  # Prevent auto-opening multiple browsers
    npm start
} -ArgumentList $ROOT_DIR

# Wait a bit for frontend to start
Start-Sleep -Seconds 5

Write-Host "`n"
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "         APPLICATION RUNNING!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "`n"

Write-Host " BACKEND API (FastAPI + ML)" -ForegroundColor Green
Write-Host "   URL: " -NoNewline
Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "   Docs: " -NoNewline
Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "`n"

Write-Host " FRONTEND UI (React + Mapbox)" -ForegroundColor Green
Write-Host "   URL: " -NoNewline
Write-Host "http://localhost:3000" -ForegroundColor Cyan
Write-Host "`n"

Write-Host " FEATURES ACTIVE:" -ForegroundColor Yellow
Write-Host "    Gemini AI (NLU)" -ForegroundColor White
Write-Host "    ML Predictions (93-96% accuracy)" -ForegroundColor White
Write-Host "    Mapbox Maps (Custom token)" -ForegroundColor White
Write-Host "    Auth0 Ready (Dev mode)" -ForegroundColor White
Write-Host "    ElevenLabs Voice" -ForegroundColor White
Write-Host "`n"

Write-Host " TIP: Opening browser to http://localhost:3000..." -ForegroundColor Cyan
Start-Sleep -Seconds 2
Start-Process "http://localhost:3000"

Write-Host "`n"
Write-Host "================================================" -ForegroundColor Magenta
Write-Host " Press Ctrl+C to stop both servers" -ForegroundColor Yellow
Write-Host "================================================" -ForegroundColor Magenta
Write-Host "`n"

# Stream output from both jobs
Write-Host " [BACKEND LOG]" -ForegroundColor Green
Write-Host " -------------" -ForegroundColor Gray
Write-Host "`n"

try {
    # Keep receiving output from both jobs
    while ($true) {
        # Backend output
        $backendOut = Receive-Job -Job $backendJob
        if ($backendOut) {
            Write-Host $backendOut -ForegroundColor Gray
        }

        # Frontend output (only errors/important messages)
        $frontendOut = Receive-Job -Job $frontendJob
        if ($frontendOut -and ($frontendOut -match "Compiled|error|Error|WARNING")) {
            Write-Host "`n [FRONTEND LOG]" -ForegroundColor Blue
            Write-Host $frontendOut -ForegroundColor Gray
        }

        # Check if jobs are still running
        if ($backendJob.State -eq "Failed" -or $backendJob.State -eq "Completed") {
            Write-Host "`n Backend stopped!" -ForegroundColor Red
            break
        }
        if ($frontendJob.State -eq "Failed" -or $frontendJob.State -eq "Completed") {
            Write-Host "`n Frontend stopped!" -ForegroundColor Red
            break
        }

        Start-Sleep -Milliseconds 500
    }
}
finally {
    # Cleanup on exit
    Write-Host "`n"
    Write-Host " Stopping servers..." -ForegroundColor Yellow
    
    Stop-Job -Job $backendJob -ErrorAction SilentlyContinue
    Stop-Job -Job $frontendJob -ErrorAction SilentlyContinue
    
    Remove-Job -Job $backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job -Job $frontendJob -Force -ErrorAction SilentlyContinue
    
    Write-Host " Servers stopped." -ForegroundColor Green
    Write-Host "`n"
}
