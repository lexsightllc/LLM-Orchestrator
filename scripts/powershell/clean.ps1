. "$PSScriptRoot/common.ps1"

$paths = @(
    'build',
    'dist',
    '.mypy_cache',
    '.pytest_cache',
    '.coverage',
    'htmlcov',
    'ruff_cache',
    'site'
)
foreach ($path in $paths) {
    $full = Join-Path $RootDir $path
    if (Test-Path $full) {
        Remove-Item -Recurse -Force $full
    }
}
