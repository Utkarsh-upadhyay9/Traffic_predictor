# SimCity AI - Quick Start Script for Windows
# This script helps you get started quickly

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SimCity AI - Quick Start Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "  ✓ Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[1/5] Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
Write-Host "  ✓ Virtual environment activated" -ForegroundColor Green

# Check if dependencies are installed
Write-Host "[3/5] Checking dependencies..." -ForegroundColor Yellow
$pipList = pip list
if ($pipList -match "fastapi") {
    Write-Host "  ✓ Dependencies already installed" -ForegroundColor Green
} else {
    Write-Host "  Installing dependencies (this may take a few minutes)..." -ForegroundColor Yellow
    cd backend
    pip install -r requirements.txt --quiet
    cd ..
    Write-Host "  ✓ Dependencies installed" -ForegroundColor Green
}

# Check .env file
Write-Host "[4/5] Checking configuration..." -ForegroundColor Yellow
if (Test-Path "backend\.env") {
    Write-Host "  ✓ Configuration file exists" -ForegroundColor Green
} else {
    Copy-Item "backend\.env.example" -Destination "backend\.env"
    Write-Host "  ⚠ Created .env file from template" -ForegroundColor Yellow
    Write-Host "  ⚠ You need to add your API keys to backend\.env" -ForegroundColor Yellow
}

# Run system tests
Write-Host "[5/5] Running system tests..." -ForegroundColor Yellow
Write-Host ""
python test_setup.py

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Edit backend\.env with your API keys" -ForegroundColor White
Write-Host "  2. Run: cd backend" -ForegroundColor White
Write-Host "  3. Run: python main.py" -ForegroundColor White
Write-Host "  4. Visit: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "For detailed instructions, see SETUP.md" -ForegroundColor Gray
Write-Host ""
