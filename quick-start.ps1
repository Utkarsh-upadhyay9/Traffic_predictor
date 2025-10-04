# SimCity AI - Quick Start Script for Windows PowerShell
# This script helps you get started quickly

Write-Host "`n"
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - Quick Start Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "`n"

# Check Python
Write-Host "[1/7] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "  ✗ Python not found! Please install Python 3.10+" -ForegroundColor Red
    exit 1
}

# Check if virtual environment exists
Write-Host "`n[2/7] Checking virtual environment..." -ForegroundColor Yellow
if (Test-Path ".\venv") {
    Write-Host "  ✓ Virtual environment exists" -ForegroundColor Green
} else {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment
Write-Host "`n[3/7] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "  ⚠ Activation may have failed - continuing anyway..." -ForegroundColor Yellow
}

# Install dependencies
Write-Host "`n[4/7] Installing Python dependencies..." -ForegroundColor Yellow
Write-Host "  (This may take 2-3 minutes)" -ForegroundColor Gray
Set-Location backend
pip install -q -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "  ✗ Failed to install dependencies" -ForegroundColor Red
    Write-Host "  Try manually: pip install -r backend/requirements.txt" -ForegroundColor Yellow
}
Set-Location ..

# Check for .env file
Write-Host "`n[5/7] Checking configuration..." -ForegroundColor Yellow
if (Test-Path ".\backend\.env") {
    Write-Host "  ✓ .env file exists" -ForegroundColor Green
} else {
    Write-Host "  Creating .env from template..." -ForegroundColor Yellow
    Copy-Item ".\backend\.env.example" ".\backend\.env"
    Write-Host "  ✓ .env file created" -ForegroundColor Green
    Write-Host "  ⚠ IMPORTANT: Edit backend\.env with your API keys!" -ForegroundColor Yellow
}

# Run tests
Write-Host "`n[6/7] Running system tests..." -ForegroundColor Yellow
Write-Host "  (This will test all integrations)" -ForegroundColor Gray
python test_setup.py
$testResult = $LASTEXITCODE

# Summary
Write-Host "`n[7/7] Setup Summary" -ForegroundColor Yellow
Write-Host "`n"
Write-Host "=========================================" -ForegroundColor Cyan

if ($testResult -eq 0) {
    Write-Host "  ✓ Setup Complete!" -ForegroundColor Green
    Write-Host "`n"
    Write-Host "  Next Steps:" -ForegroundColor Cyan
    Write-Host "  1. Edit backend\.env with your API keys" -ForegroundColor White
    Write-Host "  2. Start the backend:" -ForegroundColor White
    Write-Host "     cd backend" -ForegroundColor Gray
    Write-Host "     python main.py" -ForegroundColor Gray
    Write-Host "  3. Open http://localhost:8000/docs" -ForegroundColor White
    Write-Host "`n"
    Write-Host "  Documentation:" -ForegroundColor Cyan
    Write-Host "  - SETUP.md    - Detailed setup instructions" -ForegroundColor White
    Write-Host "  - PRIZES.md   - Prize integration details" -ForegroundColor White
    Write-Host "  - STATUS.md   - Current project status" -ForegroundColor White
} else {
    Write-Host "  ⚠ Setup completed with warnings" -ForegroundColor Yellow
    Write-Host "`n"
    Write-Host "  Some tests failed. Common issues:" -ForegroundColor Yellow
    Write-Host "  - Missing API keys in .env file (REQUIRED)" -ForegroundColor White
    Write-Host "  - MATLAB not installed (OK - mock mode works)" -ForegroundColor White
    Write-Host "  - Missing dependencies (run: pip install -r backend/requirements.txt)" -ForegroundColor White
    Write-Host "`n"
    Write-Host "  Check SETUP.md for troubleshooting" -ForegroundColor White
}

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "`n"

# Keep window open if run by double-click
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
