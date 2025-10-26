# SPDX-License-Identifier: MPL-2.0
. "$PSScriptRoot/common.ps1"

if (-not (Test-Path (Join-Path $RootDir 'sbom'))) {
    New-Item -ItemType Directory -Path (Join-Path $RootDir 'sbom') | Out-Null
}
Run-Python @('-m', 'cyclonedx_py', '--environment', 'pip', '--json', '--output', (Join-Path $RootDir 'sbom/sbom.json'))
