@echo off
echo ==========================================
echo       AI-Sentinel Launch Script
echo ==========================================

echo Activating Conda environment: microtool-ai...
call conda activate microtool-ai
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment 'microtool-ai'.
    echo Please make sure you have created the environment and conda is in your PATH.
    pause
    exit /b 1
)

echo Starting Streamlit Web UI...
streamlit run src/webui/app.py

pause
