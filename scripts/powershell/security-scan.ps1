# SPDX-License-Identifier: MPL-2.0
. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'pip_audit')
Run-Python @('-m', 'bandit', '-r', 'src/orchestrator')
Run-Python @('-m', 'detect_secrets', 'scan', '--all-files', '--exclude-files', 'sbom/.*', '--exclude-lines', 'LLM_ORCHESTRATOR_DEFAULT_MODEL') > (Join-Path $RootDir 'ci/detect-secrets-report.json')
Write-Output 'Security scan reports saved to ci/detect-secrets-report.json'
