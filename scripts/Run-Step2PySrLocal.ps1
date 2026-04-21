[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$helperPath = Join-Path $PSScriptRoot "Invoke-DeeplearningPython.ps1"
$targetScript = Join-Path $repoRoot "step2_pisr_mgn.py"

& $helperPath -ScriptPath $targetScript @ScriptArgs
exit $LASTEXITCODE
