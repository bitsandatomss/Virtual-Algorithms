"""Demand-paging algorithm package.

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

from paging_algo.paging import (  # noqa: E402
    PAGING_FEATURE_NAMES,
    POLICIES,
    Paging,
    PolicyFaultModel,
    belady_min_faults,
    collect_paging_trajectories,
    fit_learned_evictor,
    generate_refs,
    simulate_learned_eviction,
    simulate_paging,
    workload_features,
)

__all__ = [
    "PAGING_FEATURE_NAMES", "POLICIES", "Paging", "PolicyFaultModel",
    "belady_min_faults", "collect_paging_trajectories",
    "fit_learned_evictor", "generate_refs", "simulate_learned_eviction",
    "simulate_paging", "workload_features",
]
