[CmdletBinding()]
param(
    [int]$NIterations = 50,
    [int]$MaxSize = 25,
    [double]$LambdaV = 22.86045252366051
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$helperPath = Join-Path $PSScriptRoot "Invoke-DeeplearningPython.ps1"
$fitScript = Join-Path $repoRoot "step2_pisr_psi.py"
$analysisScript = Join-Path $repoRoot "step6_phi_R_analysis.py"

& $helperPath -ScriptPath $fitScript `
    --base-out-dir (Join-Path $repoRoot "outputs_mgn") `
    --input-subdir "fd_check/phi_R_input" `
    --out-subdir "fd_check/phi_R_pysr" `
    --lambda-v $LambdaV `
    --niterations $NIterations `
    --maxsize $MaxSize `
    --skip-strategy-a
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $helperPath -ScriptPath $analysisScript
exit $LASTEXITCODE
