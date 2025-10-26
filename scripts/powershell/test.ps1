. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'pytest', 'tests/unit', '--cov=orchestrator', '--cov-report=term-missing') @args
