# SimCity AI - Git Setup Script
# Initializes Git repository and prepares for first commit

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  SimCity AI - Git Repository Setup" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Git is installed
try {
    $gitVersion = git --version
    Write-Host "✓ Git is installed: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Git is not installed!" -ForegroundColor Red
    Write-Host "  Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Initialize Git repository if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "[1/4] Initializing Git repository..." -ForegroundColor Yellow
    git init
    Write-Host "  ✓ Git repository initialized" -ForegroundColor Green
} else {
    Write-Host "[1/4] Git repository already initialized" -ForegroundColor Green
}

# Add all files
Write-Host "[2/4] Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "  ✓ Files added" -ForegroundColor Green

# Create initial commit
Write-Host "[3/4] Creating initial commit..." -ForegroundColor Yellow
$commitMessage = "Initial commit: SimCity AI - Urban Traffic Simulation Platform

Project Components:
- Backend API (FastAPI) with all service integrations
- Agentuity agents (Orchestrator, Simulation, Reporting)
- MATLAB simulation scripts
- Complete documentation (README, SETUP, ARCHITECTURE, PRIZES)
- Docker configuration for Arm deployment
- Python dependencies and virtual environment setup

Prize Technologies Integrated:
✓ Google Gemini API (NLU and summarization)
✓ Auth0 (JWT authentication)
✓ Agentuity (Agent orchestration)
✓ MATLAB (Traffic simulation engine)
✓ ElevenLabs (Audio narration)
✓ Arm Architecture (Docker deployment)

Status: Ready for HackUTA 7 submission"

git commit -m $commitMessage
Write-Host "  ✓ Initial commit created" -ForegroundColor Green

# Show status
Write-Host "[4/4] Repository status:" -ForegroundColor Yellow
git status
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Git Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To push to GitHub:" -ForegroundColor White
Write-Host "  1. Create a new repository on GitHub.com" -ForegroundColor White
Write-Host "  2. Run: git remote add origin <your-repo-url>" -ForegroundColor White
Write-Host "  3. Run: git branch -M main" -ForegroundColor White
Write-Host "  4. Run: git push -u origin main" -ForegroundColor White
Write-Host ""
Write-Host "Repository is ready!" -ForegroundColor Green
Write-Host ""
