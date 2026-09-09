"""Virtual Algorithms: learned virtualizations of distributed algorithms.

Implements the context.txt thesis:

  physical / real system -> learned computational equivalent

A VirtualAlgorithm is NOT a simulator (Rules -> state -> next state).
It is a surrogate (Observations -> learned latent structure -> predicted
next observation) exposing its own virtual state, operations, invariants
and semantics -- cf. virtual memory vs physical memory -- while a TrustGate
(calibration, conservation constraints, OOD detection, uncertainty
propagation, active selection) decides when virtual experimentation is
trustworthy.
"""

from rlraft.virtual.base import (
    Intervention,
    MaturityLevel,
    Prediction,
    TrustReport,
    VirtualAlgorithm,
    VirtualState,
    VIRTUAL_ALGORITHM_REGISTRY,
    register_virtual_algorithm,
)
# Sibling algo packages live next to "Raft Algorithm" (TCP/Routing/Paging/
# Scheduling Algorithm). ensure_algo_paths() is CWD-independent (anchored
# at this file), so these imports work from any working directory.
from rlraft.virtual.paths import ensure_algo_paths as _ensure_algo_paths

_ensure_algo_paths()

from rlraft.virtual.invariants import (
    BackoffInvariants,
    GossipInvariants,
    PagingInvariants,
    RaftInvariants,
    RoutingInvariants,
    SchedInvariants,
    TCPInvariants,
)
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker, MahalanobisOOD, reliability_table, wilson_interval
from rlraft.virtual.surrogate import DynamicsSurrogate, EnsembleDynamicsSurrogate, EnsembleRegressor, TrajectoryDataset
from rlraft.virtual import eval as eval_harness
from rlraft.virtual.algorithms import (
    Backoff,
    Gossip,
    RaftElection,
)
# Sibling algo classes load lazily (PEP 562): importing them eagerly here
# would cycle back through the algo packages' own __init__ (which imports
# rlraft.virtual.*). Direct imports (tcp_algo.tcp, ...) always work.
_LAZY_ALGOS = {
    "TCPCongestion": "tcp_algo.tcp",
    "OverlayRouting": "routing_algo.routing",
    "Paging": "paging_algo.paging",
    "Scheduling": "sched_algo.sched",
}


def __getattr__(name: str):
    if name in _LAZY_ALGOS:
        import importlib

        module = importlib.import_module(_LAZY_ALGOS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from rlraft.virtual.lab import VirtualLab, LabRoundResult
from rlraft.virtual import cross_audit as cross_audit_harness

__all__ = [
    "Intervention",
    "MaturityLevel",
    "Prediction",
    "TrustReport",
    "VirtualAlgorithm",
    "VirtualState",
    "VIRTUAL_ALGORITHM_REGISTRY",
    "register_virtual_algorithm",
    "RaftInvariants",
    "BackoffInvariants",
    "GossipInvariants",
    "TCPInvariants",
    "RoutingInvariants",
    "PagingInvariants",
    "SchedInvariants",
    "Calibrator",
    "FeatureDistribution",
    "MahalanobisOOD",
    "reliability_table",
    "wilson_interval",
    "TrustGate",
    "UncertaintyTracker",
    "DynamicsSurrogate",
    "EnsembleDynamicsSurrogate",
    "EnsembleRegressor",
    "TrajectoryDataset",
    "eval_harness",
    "Backoff",
    "Gossip",
    "RaftElection",
    "TCPCongestion",
    "OverlayRouting",
    "Paging",
    "Scheduling",
    "VirtualLab",
    "LabRoundResult",
    "cross_audit_harness",
]
