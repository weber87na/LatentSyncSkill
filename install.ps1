# Run from any directory: powershell -ExecutionPolicy Bypass -File .\install.ps1
$ErrorActionPreference = 'Stop'
$installerPath = Join-Path $PSScriptRoot 'scripts\install.py'
if (Get-Command python -ErrorAction SilentlyContinue) {
    & python $installerPath @args
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $installerPath @args
} else {
    Write-Error 'Python 3.10+ is required. Install Python, then rerun this script.'
    exit 1
}
exit $LASTEXITCODE
