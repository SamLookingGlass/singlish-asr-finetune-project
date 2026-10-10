# Create .venv and install the project plus a CUDA build of PyTorch.
#
# The RTX 50-series (Blackwell, sm_120) needs a cu128 wheel. A plain
# `pip install torch` often installs an older CUDA build that cannot
# see that GPU. This script installs the project, then puts the cu128
# wheel back so it is the one that stays.
#
# Usage (from the repo root or from anywhere):
#   .\scripts\setup.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path .venv)) {
    Write-Host "Creating .venv"
    python -m venv .venv
}

$python = ".venv\Scripts\python.exe"
& $python -m pip install -U pip
& $python -m pip install -e ".[dev]"
& $python -m pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128

& $python -c "import torch; name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'; print(torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), name)"
