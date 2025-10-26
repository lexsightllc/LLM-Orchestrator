. "$PSScriptRoot/common.ps1"

Ensure-Venv
Run-Pip @('install', '--upgrade', 'pip')
Run-Pip @('install', '-r', (Join-Path $RootDir 'requirements.txt'))
Run-Pip @('install', '-r', (Join-Path $RootDir 'requirements-dev.txt'))
Run-Python @('-m', 'pre_commit', 'install', '--install-hooks')
Run-Python @('-m', 'pre_commit', 'install', '--hook-type', 'commit-msg')
Write-Output "Environment bootstrapped. Activate with: .venv\\Scripts\\Activate.ps1"
