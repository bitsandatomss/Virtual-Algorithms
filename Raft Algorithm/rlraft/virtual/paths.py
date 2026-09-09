"""Sibling algo-folder bootstrap (CWD-independent).

The four algo packages (TCP/Routing/Paging/Scheduling Algorithm) live next
to "Raft Algorithm" under the workspace root, not inside the rlraft
package. This helper puts those folders on sys.path using paths anchored
at THIS file, so imports work no matter which directory tests or tools
run from. Appends only -- never shadows anything already importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ALGO_FOLDERS = (
    "TCP Algorithm",
    "Routing Algorithm",
    "Paging Algorithm",
    "Scheduling Algorithm",
)


def workspace_root() -> Path:
    # .../Raft Algorithm/rlraft/virtual/paths.py -> up 4 = workspace root
    return Path(__file__).resolve().parent.parent.parent.parent


def ensure_algo_paths() -> list[str]:
    added: list[str] = []
    root = workspace_root()
    for name in ALGO_FOLDERS:
        d = root / name
        s = str(d)
        if d.is_dir() and s not in sys.path:
            sys.path.append(s)
            added.append(s)
    return added
