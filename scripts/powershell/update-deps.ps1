. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'piptools', 'compile', (Join-Path $RootDir 'requirements.in'), '--output-file', (Join-Path $RootDir 'requirements.txt')) @args
Run-Python @('-m', 'piptools', 'compile', (Join-Path $RootDir 'requirements-dev.in'), '--output-file', (Join-Path $RootDir 'requirements-dev.txt')) @args
