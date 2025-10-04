#!/usr/bin/env pwsh
# SimCity AI - Unified Startup Script
# Starts both backend and frontend in one terminal

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - Starting Application" -ForegroundColor Green  
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment and set env vars
Write-Host "Activating Python environment..." -ForegroundColor Yellow
. .\venv\Scripts\Activate.ps1
$env:SKIP_AUTH_VERIFICATION = "true"

Write-Host ""
Write-Host "Starting Backend Server (Port 8000)..." -ForegroundColor Cyan

# Start backend in a new minimized window
Start-Process powershell -ArgumentList @(
    "-NoExit"
    "-Command"
    "cd '$PWD'; . .\venv\Scripts\Activate.ps1; `$env:SKIP_AUTH_VERIFICATION='true'; Write-Host ''; Write-Host 'BACKEND SERVER' -ForegroundColor Green; Write-Host '   http://localhost:8000' -ForegroundColor Cyan; Write-Host ''; python backend/main.py"
) -WindowStyle Normal

Write-Host "Backend starting in separate window..." -ForegroundColor Green
Write-Host ""
Write-Host "Waiting 8 seconds for backend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "   SERVERS RUNNING" -ForegroundColor White
Write-Host "================================================" -ForegroundColor Green  
Write-Host ""
Write-Host "  Backend API:  " -NoNewline -ForegroundColor Cyan
Write-Host "http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs:     " -NoNewline -ForegroundColor Cyan
Write-Host "http://localhost:8000/docs" -ForegroundColor White
Write-Host "  Frontend UI:  " -NoNewline -ForegroundColor Cyan  
Write-Host "http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Starting Frontend (React)..." -ForegroundColor Cyan
Write-Host "   (This terminal will show frontend logs)" -ForegroundColor Gray
Write-Host ""
Write-Host "   Press Ctrl+C here to stop ONLY frontend" -ForegroundColor Yellow
Write-Host "   Close the backend window to stop backend" -ForegroundColor Yellow
Write-Host ""

# Change to frontend and start (this keeps terminal active)
Set-Location frontend
npm start
