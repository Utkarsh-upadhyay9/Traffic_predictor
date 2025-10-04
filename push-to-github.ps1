# SimCity AI - GitHub Setup Helper
# Run this script after creating your GitHub repository

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "   SimCity AI - GitHub Push Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Get repository URL
Write-Host "Enter your GitHub repository URL:" -ForegroundColor Yellow
Write-Host "Example: https://github.com/Utkarsh-upadhyay9/simcity-ai.git" -ForegroundColor Gray
$repoUrl = Read-Host "Repository URL"

if ([string]::IsNullOrWhiteSpace($repoUrl)) {
    Write-Host "❌ No URL provided. Using default..." -ForegroundColor Red
    $repoUrl = "https://github.com/Utkarsh-upadhyay9/simcity-ai.git"
}

Write-Host ""
Write-Host "📋 Repository URL: $repoUrl" -ForegroundColor Green
Write-Host ""

# Check if remote already exists
$remoteExists = git remote get-url origin 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "⚠️  Remote 'origin' already exists. Removing..." -ForegroundColor Yellow
    git remote remove origin
}

# Add remote
Write-Host "➕ Adding remote 'origin'..." -ForegroundColor Cyan
git remote add origin $repoUrl

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Remote added successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Failed to add remote" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Rename branch to main
Write-Host "🔄 Renaming branch to 'main'..." -ForegroundColor Cyan
git branch -M main

# Push to GitHub
Write-Host ""
Write-Host "🚀 Pushing to GitHub..." -ForegroundColor Cyan
Write-Host "   This may take a minute..." -ForegroundColor Gray
Write-Host ""

git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "================================================" -ForegroundColor Green
    Write-Host "   ✅ Successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "🎉 Your project is now on GitHub!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "View it at:" -ForegroundColor Yellow
    $repoUrlWeb = $repoUrl -replace '\.git$', ''
    Write-Host $repoUrlWeb -ForegroundColor White
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Get Gemini API key: https://makersuite.google.com/app/apikey" -ForegroundColor Gray
    Write-Host "  2. Create .env file: cd backend && cp .env.example .env" -ForegroundColor Gray
    Write-Host "  3. Test the API: python main.py" -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Push failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Possible reasons:" -ForegroundColor Yellow
    Write-Host "  1. Repository doesn't exist on GitHub yet" -ForegroundColor Gray
    Write-Host "  2. Authentication failed (set up Git credentials)" -ForegroundColor Gray
    Write-Host "  3. Network connection issue" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Try:" -ForegroundColor Cyan
    Write-Host "  1. Create the repository on GitHub first" -ForegroundColor Gray
    Write-Host "  2. Run this script again" -ForegroundColor Gray
    Write-Host ""
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
