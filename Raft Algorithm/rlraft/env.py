from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str | None = None, override: bool = False) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from .env without printing secrets."""

    env_path = Path(path) if path else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded
