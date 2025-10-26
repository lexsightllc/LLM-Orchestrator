import sys
from pathlib import Path

# Ensure repository root and src directory are on sys.path for tests executing from subdirectories
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

for path in (ROOT, SRC):
    str_path = str(path)
    if str_path not in sys.path:
        sys.path.insert(0, str_path)
