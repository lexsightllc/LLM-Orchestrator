. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'build', '--wheel')
