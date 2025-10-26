# SPDX-License-Identifier: MPL-2.0
. "$PSScriptRoot/common.ps1"

Run-Python @('-m', 'mypy', 'src/orchestrator')
