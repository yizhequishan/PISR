[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = "D:\keyan\Anaconda\envs\deeplearning\python.exe"
$juliaProject = "D:\keyan\Anaconda\julia_env"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python interpreter not found: $pythonExe"
}

if (-not (Test-Path -LiteralPath $juliaProject)) {
    throw "Julia project not found: $juliaProject"
}

$resolvedScript = (Resolve-Path -LiteralPath $ScriptPath).Path

$env:PYTHON_JULIACALL_HANDLE_SIGNALS = "yes"
$env:PYTHON_JULIACALL_THREADS = "auto"
$env:PYTHON_JULIACALL_OPTLEVEL = "3"
$env:JULIA_PROJECT = $juliaProject
$env:PYTHONUTF8 = "1"

Push-Location $repoRoot
try {
    & $pythonExe $resolvedScript @ScriptArgs
    $exitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($null -ne $exitCode -and $exitCode -ne 0) {
    exit $exitCode
}
