@echo off
setlocal

rem Run this file from Anaconda Prompt on a new Windows computer.
cd /d "%~dp0"

where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda was not found.
    echo Open Anaconda Prompt and run this script again.
    exit /b 1
)

echo [1/4] Creating the mico-pminet Conda environment...
call conda env create -f environment.yml
if errorlevel 1 (
    echo.
    echo [ERROR] Environment creation failed.
    echo If mico-pminet already exists, use: conda activate mico-pminet
    exit /b 1
)

echo [2/4] Activating the environment and installing this repository...
call conda activate mico-pminet
if errorlevel 1 exit /b 1
python -m pip install --no-deps -e .
if errorlevel 1 exit /b 1

echo [3/4] Running automated tests...
python -m pytest -q
if errorlevel 1 exit /b 1

echo [4/4] Verifying released data, checkpoint, and numerical results...
python run.py verify
if errorlevel 1 exit /b 1

echo.
echo [SUCCESS] The reproducibility environment is ready.
echo For later sessions: conda activate mico-pminet
endlocal
