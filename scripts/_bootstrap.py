from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

for path in (str(SRC), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
