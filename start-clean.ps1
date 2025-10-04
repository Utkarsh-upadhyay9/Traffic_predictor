#!/usr/bin/env pwsh
# SimCity AI - Clean Start Script
# Closes everything and starts fresh

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - Clean Startup" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill all existing processes
Write-Host "[1/3] Stopping all existing servers..." -ForegroundColor Yellow
Get-Process python,node -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Write-Host "      All processes stopped." -ForegroundColor Green
Write-Host ""

# Step 2: Start backend
Write-Host "[2/3] Starting Backend API..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1
$env:SKIP_AUTH_VERIFICATION = "true"

Start-Process powershell -ArgumentList @(
    "-NoExit"
    "-Command"
    "cd '$PWD'; . .\venv\Scripts\Activate.ps1; `$env:SKIP_AUTH_VERIFICATION='true'; Write-Host ''; Write-Host '==================================' -ForegroundColor Cyan; Write-Host '  BACKEND SERVER' -ForegroundColor Green; Write-Host '  http://localhost:8000' -ForegroundColor White; Write-Host '==================================' -ForegroundColor Cyan; Write-Host ''; python backend/main.py"
) -WindowStyle Normal

Write-Host "      Backend starting..." -ForegroundColor Green
Start-Sleep -Seconds 8

# Step 3: Open the simple HTML page
Write-Host "[3/3] Opening Traffic Prediction Page..." -ForegroundColor Yellow
$htmlPath = Join-Path $PWD "index.html"
Start-Process $htmlPath

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "   READY!" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host "  Web Page:     Opened in browser" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C here to exit" -ForegroundColor Yellow
Write-Host "  Close backend window to stop server" -ForegroundColor Yellow
Write-Host ""

# Keep this terminal open
Write-Host "Press any key to stop and exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

# Cleanup on exit
Write-Host ""
Write-Host "Stopping all servers..." -ForegroundColor Yellow
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "Done." -ForegroundColor Green
