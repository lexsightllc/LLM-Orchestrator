$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

& (Join-Path $root 'scripts/powershell/fmt.ps1') -Check
& (Join-Path $root 'scripts/powershell/lint.ps1')
& (Join-Path $root 'scripts/powershell/typecheck.ps1')
& (Join-Path $root 'scripts/powershell/test.ps1')
& (Join-Path $root 'scripts/powershell/coverage.ps1')
& (Join-Path $root 'scripts/powershell/security-scan.ps1')
