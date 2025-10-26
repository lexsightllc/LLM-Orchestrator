$Script:RootDir = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$Script:VenvDir = Join-Path $RootDir '.venv'
$Script:PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { Join-Path $VenvDir 'Scripts\python.exe' }
$Script:PipBin = if ($env:PIP_BIN) { $env:PIP_BIN } else { Join-Path $VenvDir 'Scripts\pip.exe' }

function Ensure-Venv {
    if (Test-Path $VenvDir -PathType Container -and (Test-Path $PythonBin)) {
        return
    }
    python -m venv $VenvDir
    & (Join-Path $VenvDir 'Scripts\pip.exe') install --upgrade pip
}

function Run-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    Ensure-Venv
    & $PythonBin @Arguments
}

function Run-Pip {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    Ensure-Venv
    & $PipBin @Arguments
}
