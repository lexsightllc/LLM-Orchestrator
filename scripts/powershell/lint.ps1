# SPDX-License-Identifier: MPL-2.0
param(
    [switch]$Fix
)

. "$PSScriptRoot/common.ps1"

$ruffArgs = @('check', 'src', 'tests')
if ($Fix) {
    $ruffArgs += '--fix'
}
Run-Python @('-m', 'ruff') @ruffArgs
Run-Python @('-m', 'bandit', '-r', 'src/orchestrator')
