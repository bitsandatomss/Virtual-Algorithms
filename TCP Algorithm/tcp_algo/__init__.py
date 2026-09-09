"""TCP congestion-control algorithm package.

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

from tcp_algo.tcp import (  # noqa: E402
    CAPS,
    MSS_BYTES,
    TCP_FEATURE_NAMES,
    AimdThroughputModel,
    TCPCongestion,
    collect_tcp_trajectories,
    simulate_aimd,
    tcp_features,
)

__all__ = [
    "CAPS", "MSS_BYTES", "TCP_FEATURE_NAMES", "AimdThroughputModel",
    "TCPCongestion", "collect_tcp_trajectories", "simulate_aimd",
    "tcp_features",
]
