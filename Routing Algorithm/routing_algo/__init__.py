"""Overlay-routing algorithm package.

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

from routing_algo.routing import (  # noqa: E402
    ROUTING_FEATURE_NAMES,
    RoutingSurrogate,
    OverlayRouting,
    build_topology,
    collect_routing_trajectories,
    enumerate_paths,
    path_link_params,
    routing_features,
    simulate_path,
)

__all__ = [
    "ROUTING_FEATURE_NAMES", "RoutingSurrogate", "OverlayRouting",
    "build_topology", "collect_routing_trajectories", "enumerate_paths",
    "path_link_params", "routing_features", "simulate_path",
]
