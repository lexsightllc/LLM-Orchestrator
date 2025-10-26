# SPDX-License-Identifier: MPL-2.0
param(
    [switch]$Check
)

. "$PSScriptRoot/common.ps1"

if ($Check) {
    Run-Python @('-m', 'black', '--check', 'src', 'tests')
    Run-Python @('-m', 'isort', '--check-only', 'src', 'tests')
    Run-Python @('-m', 'ruff', 'format', '--check', 'src', 'tests')
} else {
    Run-Python @('-m', 'black', 'src', 'tests')
    Run-Python @('-m', 'isort', 'src', 'tests')
    Run-Python @('-m', 'ruff', 'format', 'src', 'tests')
}
