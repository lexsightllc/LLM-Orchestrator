# SPDX-License-Identifier: MPL-2.0
. "$PSScriptRoot/common.ps1"

$e2eDir = Join-Path $RootDir 'tests/e2e'
$hasTests = Test-Path $e2eDir -PathType Container -and (Get-ChildItem $e2eDir -Filter 'test_*.py' -Recurse | Select-Object -First 1)
if (-not $hasTests) {
    Write-Warning 'No end-to-end tests detected. Skipping.'
    exit 0
}

Run-Python @('-m', 'pytest', 'tests/e2e') @args
