param([Parameter(ValueFromRemainingArguments = $true)][string[]]$TestPaths)
$ErrorActionPreference = 'Stop'
if (-not $TestPaths) { Write-Output 'usage: repo-tests.ps1 <test path> [more paths]'; exit 2 }
foreach ($path in $TestPaths) {
    if ($path -notmatch '^[A-Za-z0-9_./-]+$') { throw "invalid test path: $path" }
}
$root = Join-Path $PSScriptRoot 'workspace'
$inner = 'PYTHONPATH=/app/lib:/app/test/lib python -m pytest ' + ($TestPaths -join ' ') + ' -v --tb=line -p no:cacheprovider'
docker run --rm --network none -v "${root}:/app" -w /app __IMAGE__ bash -lc $inner
exit $LASTEXITCODE
