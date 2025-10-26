. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'mypy', 'src/orchestrator')
