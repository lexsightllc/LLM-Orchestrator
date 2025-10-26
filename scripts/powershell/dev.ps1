. "$PSScriptRoot/common.ps1"

if ($env:WATCH -eq '1') {
    $script = @'
import subprocess
import sys
from watchfiles import run_process

CMD = [sys.executable, '-m', 'pytest', 'tests/unit', '--maxfail=1', '--lf']

def _runner():
    result = subprocess.run(CMD, check=False)
    if result.returncode != 0:
        print(f"Test run exited with {result.returncode}", file=sys.stderr)

run_process(['src', 'tests'], target=_runner)
'@
    Run-Python @('-c', $script)
} else {
    Run-Python @('-m', 'pytest', 'tests/unit', '--maxfail=1', '--lf')
}
