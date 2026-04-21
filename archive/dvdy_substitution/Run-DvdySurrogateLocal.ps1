[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Archived: this experiment was rejected as a main-flow method
# (see ../../报告.md section on "B+g substitution 失败记录").
# Paths updated after moving to archive/dvdy_substitution/.
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$helperPath = Join-Path $repoRoot "scripts/Invoke-DeeplearningPython.ps1"
$targetScript = Join-Path $PSScriptRoot "experiment_dvdy_surrogate.py"

& $helperPath -ScriptPath $targetScript @ScriptArgs
exit $LASTEXITCODE
