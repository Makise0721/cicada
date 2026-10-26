param([Parameter(ValueFromRemainingArguments = $true)][string[]]$TestPaths)
$ErrorActionPreference = 'Stop'
if (-not $TestPaths) { Write-Output 'usage: repo-tests.ps1 <test path> [more paths]'; exit 2 }
& '__PYTHON__' -m tests.support.swebench_batch repo-test --endpoint '__ENDPOINT__' --token '__TOKEN__' -- @TestPaths
exit $LASTEXITCODE
