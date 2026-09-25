# Setup script for Windows (PowerShell)
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "Setting up Business Entity Resolution Environment" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

python -m venv .venv
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to create virtual environment. Ensure Python 3.10+ is available in PATH."
    exit 1
}

$venvPython = ".\.venv\Scripts\python.exe"
$venvPip    = ".\.venv\Scripts\pip.exe"
$venvPytest = ".\.venv\Scripts\pytest.exe"

& $venvPip install --upgrade pip
& $venvPip install -r requirements-lock.txt

Write-Host "`nRunning test suite to verify setup..." -ForegroundColor Yellow
& $venvPytest tests/ -v

Write-Host "`nVirtual environment setup complete!" -ForegroundColor Green
Write-Host "To activate in PowerShell: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "To run pipeline: & .\.venv\Scripts\python.exe run_pipeline.py --mode all" -ForegroundColor Green
