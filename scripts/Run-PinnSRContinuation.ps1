[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutDir,
    [string]$Checkpoint = "outputs_pinn/cylinder/hard_boundary_refined_20261013/cylinder_poiseuille/checkpoint_last.pt",
    [double]$TotalSeconds = 300,
    [double]$SearchSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\keyan\Anaconda\envs\deeplearning\python.exe"
$juliaProject = "D:\keyan\Anaconda\julia_env"
$juliaExe = Join-Path $juliaProject "pyjuliapkg\install\bin\julia.exe"
if ($TotalSeconds -le 0 -or $SearchSeconds -le 0) { throw "Time budgets must be positive" }
if (-not (Test-Path -LiteralPath $pythonExe) -or -not (Test-Path -LiteralPath $juliaExe)) {
    throw "The configured local Python/Julia installation is missing"
}
$env:MKL_THREADING_LAYER = "SEQUENTIAL"
$env:PYTHONUTF8 = "1"
$env:PYTHON_JULIACALL_HANDLE_SIGNALS = "yes"
$env:PYTHON_JULIACALL_THREADS = "auto"
$env:PYTHON_JULIACALL_OPTLEVEL = "3"
$env:JULIA_PROJECT = $juliaProject
$env:PYTHON_JULIAPKG_PROJECT = $juliaProject
$env:PYTHON_JULIAPKG_EXE = $juliaExe
$env:PYTHON_JULIAPKG_OFFLINE = "yes"
Push-Location $repoRoot
try {
    & $pythonExe scripts/pinn_sr_residual_trial.py prepare --checkpoint $Checkpoint --out-dir $OutDir --search-seconds $SearchSeconds
    if ($LASTEXITCODE -ne 0) { throw "Symbolic preparation failed; no continuation launched" }
    foreach ($kind in @("ordinary", "inlet", "sr")) {
        & $pythonExe scripts/pinn_sr_residual_trial.py run --out-dir $OutDir --kind $kind --total-seconds $TotalSeconds
        if ($LASTEXITCODE -ne 0) { throw "Continuation failed: $kind" }
    }
    & $pythonExe scripts/pinn_sr_residual_trial.py summarize --out-dir $OutDir
    if ($LASTEXITCODE -ne 0) { throw "Final audit failed" }
    & $pythonExe scripts/plot_pinn_sr_residual_trial.py --out-dir $OutDir
    if ($LASTEXITCODE -ne 0) { throw "Plot generation failed" }
}
finally { Pop-Location }
