. "$PSScriptRoot/common.ps1"

$migrationsDir = Join-Path $RootDir 'infra/migrations'
if (-not (Test-Path $migrationsDir -PathType Container)) {
    Write-Warning "No migrations directory found at $migrationsDir."
    exit 0
}

Write-Warning 'Implement database migration runner for your chosen backend.'
