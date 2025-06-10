@echo off
echo 🚀 Starting Comprehensive Multi-Page Performance Analysis
echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call myVenv\Scripts\activate.bat

REM Check if activation was successful
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    echo Please make sure myVenv exists and is properly configured
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Run the analysis
echo 📊 Running comprehensive analysis...
echo.
python run_analysis.py

REM Check if analysis was successful
if errorlevel 1 (
    echo.
    echo ❌ Analysis failed with errors
    pause
    exit /b 1
)

echo.
echo 🎉 Analysis completed successfully!
echo 📁 Check the output/ folder for results
echo.
pause
