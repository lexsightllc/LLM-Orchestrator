import sys
from pathlib import Path

# Ensure repository root is on sys.path for tests executing from subdirectories
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
