# PowerShell script to run comprehensive analysis
Write-Host "🚀 Starting Comprehensive Multi-Page Performance Analysis" -ForegroundColor Green
Write-Host ""

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
try {
    & ".\myVenv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Please make sure myVenv exists and is properly configured" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Run the analysis
Write-Host "📊 Running comprehensive analysis..." -ForegroundColor Yellow
Write-Host ""

try {
    python run_analysis.py
    Write-Host ""
    Write-Host "🎉 Analysis completed successfully!" -ForegroundColor Green
    Write-Host "📁 Check the output/ folder for results" -ForegroundColor Cyan
} catch {
    Write-Host ""
    Write-Host "❌ Analysis failed with errors" -ForegroundColor Red
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"
