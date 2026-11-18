param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 8000,
    [string]$Partition = "student_test",
    [string]$EvaluatorDirectory = ""
)

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = Join-Path $repoRoot "src"
$env:PATHSNAP_SERVE_PARTITION = $Partition
if ($EvaluatorDirectory) {
    $env:PATHSNAP_EVALUATOR_DIR = (Resolve-Path -LiteralPath $EvaluatorDirectory).Path
}

& (Join-Path $repoRoot ".venv\Scripts\python.exe") -m uvicorn pathsnap.api:app --host $HostAddress --port $Port
