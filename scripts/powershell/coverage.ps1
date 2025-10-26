. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'pytest', 'tests', '--cov=orchestrator', '--cov-report=term-missing', '--cov-report=xml')
Run-Python @('-m', 'coverage', 'report', '--fail-under=85')
