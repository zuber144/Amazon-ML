@echo off
REM Setup script for Windows (Command Prompt)
echo ====================================================
echo Setting up Business Entity Resolution Environment
echo ====================================================

python -m venv .venv
if %errorlevel% neq 0 (
    echo Error creating virtual environment. Ensure Python 3.10+ is installed.
    exit /b %errorlevel%
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements-lock.txt

echo.
echo ====================================================
echo Running test suite to verify setup...
echo ====================================================
pytest tests/ -v

echo.
echo Virtual environment setup complete!
echo To activate in Command Prompt: call .venv\Scripts\activate.bat
echo To run pipeline: python run_pipeline.py --mode all
