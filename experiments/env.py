"""Load the project's .env into os.environ without overriding what is already set; no dependency.

    import sys; sys.path.insert(0, "experiments"); import env; env.load()
"""

import os
from pathlib import Path


def find(start=None):
    """The nearest .env at or above `start` (default: this file's project root)."""
    for d in [Path(start or Path(__file__).resolve().parent).resolve(), *Path(start or __file__).resolve().parents]:
        if (d / ".env").is_file():
            return d / ".env"
    return None


def load(path=None):
    """Set each KEY=value of the .env that is not already set and not empty; returns the keys it set."""
    path = Path(path) if path else find()
    if not path:
        return []
    set_ = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = (s.strip() for s in line.removeprefix("export ").split("=", 1))
        v = v.strip("'\"")
        if k and v and k not in os.environ:
            os.environ[k] = v
            set_.append(k)
    return set_
