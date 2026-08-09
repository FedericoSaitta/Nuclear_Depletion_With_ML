# One-time setup on the desktop. ML environment only — see OUTPUT.md §5.4 for
# why OpenMC on Windows should go through WSL2.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

uv python install                      # honours .python-version -> CPython 3.12
uv sync --extra ml --locked            # ~3 GB of CUDA wheels on first run

uv run python -c @"
import torch, lightning, torchdiffeq
print(f'torch      {torch.__version__}')
print(f'lightning  {lightning.__version__}')
print(f'cuda       {torch.cuda.is_available()}  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else """"}')
"@

Write-Host ""
Write-Host "Ready.  Train with:  uv run nucml --config ML/main_config.yaml"
