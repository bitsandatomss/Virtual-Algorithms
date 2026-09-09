"""CPU/job-scheduling algorithm package.

Runnable standalone (imports rlraft shared core via path anchored at this
file) or through the Raft Algorithm suite (rlraft.virtual ensures paths).
"""
import sys as _sys
from pathlib import Path as _Path

for _p in (_Path(__file__).resolve().parent.parent,
           _Path(__file__).resolve().parent.parent.parent / "Raft Algorithm"):
    _s = str(_p)
    if _s not in _sys.path:
        _sys.path.append(_s)
del _sys, _Path

from sched_algo.sched import (  # noqa: E402
    DISCIPLINES,
    SCHED_FEATURE_NAMES,
    SchedWaitModel,
    Scheduling,
    collect_sched_trajectories,
    generate_jobs,
    sched_features,
    simulate_sched,
)

__all__ = [
    "DISCIPLINES", "SCHED_FEATURE_NAMES", "SchedWaitModel", "Scheduling",
    "collect_sched_trajectories", "generate_jobs", "sched_features",
    "simulate_sched",
]
