# SPDX-License-Identifier: MPL-2.0
. "$PSScriptRoot/common.ps1"

& (Join-Path $RootDir 'scripts/powershell/check.ps1')
Run-Python @('-m', 'build')
Run-Python @('-m', 'twine', 'check', 'dist/*')
Write-Output 'Artifacts ready in dist/. Tag and publish using your release workflow.'
