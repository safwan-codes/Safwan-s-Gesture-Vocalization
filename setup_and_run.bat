@echo off
setlocal
echo ========================================================
echo Gesture Vocalization Setup and Run Script
echo ========================================================
echo.

REM Check if Git is installed
git --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Git is not installed or not in PATH. Please install Git.
    pause
    exit /b
)

REM Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed or not in PATH. Please install Python 3.8+.
    pause
    exit /b
)

REM Define the target directory name
set REPO_DIR=Safwan-s-Gesture-Vocalization

REM Clone the repository if it doesn't exist in the current folder
IF NOT EXIST "%REPO_DIR%" (
    echo [INFO] Cloning the repository from GitHub...
    git clone https://github.com/safwan-codes/Safwan-s-Gesture-Vocalization.git
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to clone the repository.
        pause
        exit /b
    )
) ELSE (
    echo [INFO] Found existing folder "%REPO_DIR%". Pulling latest changes...
    cd "%REPO_DIR%"
    git pull origin main
    cd ..
)

cd "%REPO_DIR%"

REM Create a virtual environment if it doesn't exist
IF NOT EXIST "venv" (
    echo [INFO] Creating an isolated Python virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Navigate to Source Code directory
cd "Source Code"

REM Install requirements
echo [INFO] Installing required libraries...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt

REM Start the application
echo [INFO] Starting the Gesture Vocalization Dashboard...
echo [INFO] Your browser will open shortly...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

python app.py

pause
