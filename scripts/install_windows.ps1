param(
    [string]$WheelPath = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"

if (-not $WheelPath) {
    $Wheel = Get-ChildItem -Path $Root -Filter "hero_siege_bot-*.whl" |
        Select-Object -First 1
    if (-not $Wheel) {
        $Wheel = Get-ChildItem -Path (Join-Path $Root "dist") `
            -Filter "hero_siege_bot-*.whl" -ErrorAction SilentlyContinue |
            Select-Object -First 1
    }
    if (-not $Wheel) {
        throw "Wheel not found. Put hero_siege_bot-*.whl beside this release bundle."
    }
    $WheelPath = $Wheel.FullName
}

Write-Host "Creating Python 3.12 environment..."
py -3.12 -m venv $Venv
$Python = Join-Path $Venv "Scripts\python.exe"

Write-Host "Installing $WheelPath..."
& $Python -m pip install --upgrade pip
& $Python -m pip install $WheelPath

Write-Host ""
Write-Host "Installed. First run the safe capture command:"
Write-Host "  .\.venv\Scripts\python.exe scripts\collect_frames.py --count 10"
Write-Host ""
Write-Host "Then follow README.md and docs\windows-smoke-test.md."
