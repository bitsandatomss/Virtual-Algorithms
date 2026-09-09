"""Core abstraction for virtual algorithms.

context.txt mapping:
  - virtualization = computationally accessible representation grounded in,
    but not identical to, physical realization (lines ~1018-1023)
  - virtual object has own state, operations, invariants, semantics (~1117)
  - surrogate vs simulation: Observations -> latent -> predicted obs (~352-356)
  - maturity L1..L5: exemplar -> interpolative -> counterfactual ->
    mechanistic -> scientific (~664-682)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class MaturityLevel(IntEnum):
    EXEMPLAR = 1  # reproduces previously observed trajectories
    INTERPOLATIVE = 2  # unseen combinations of known conditions
    COUNTERFACTUAL = 3  # useful predictions under interventions
    MECHANISTIC = 4  # latent structure generalizes OOD
    SCIENTIFIC = 5  # proposes experiments, quantifies uncertainty, active loop


@dataclass(slots=True)
class VirtualState:
    """Effective computational object the agent operates on.

    Deliberately smaller than the physical realization: it preserves only
    the distinctions relevant to the task family (conditions of
    intelligibility), exactly like a virtual address space preserves only
    what a process needs.
    """

    features: list[float]
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(zip(self.feature_names, self.features))


@dataclass(slots=True)
class Intervention:
    """A counterfactual action applied to the virtual object.

    e.g. raft: force timeout arm; backoff: force base delay;
    gossip: force fanout. The surrogate must predict the outcome
    without executing the physical system.
    """

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Prediction:
    outcome: dict[str, float]
    uncertainty: float  # std / entropy-scale, must grow under OOD & long horizon
    ood_score: float = 0.0
    trusted: bool = False


@dataclass(slots=True)
class TrustReport:
    trusted: bool
    reasons: list[str] = field(default_factory=list)
    ood_score: float = 0.0
    calibration_error: float = 0.0
    uncertainty: float = 0.0
    maturity: MaturityLevel = MaturityLevel.EXEMPLAR
    invariant_violations: list[str] = field(default_factory=list)


VIRTUAL_ALGORITHM_REGISTRY: dict[str, type["VirtualAlgorithm"]] = {}


def register_virtual_algorithm(name: str):
    def decorator(cls: type["VirtualAlgorithm"]) -> type["VirtualAlgorithm"]:
        VIRTUAL_ALGORITHM_REGISTRY[name] = cls
        cls.algorithm_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


class VirtualAlgorithm(ABC):
    """Physical algorithm -> learned computational equivalent.

    Subclasses must define:
      - physical_step: explicit-rules execution (ground truth)
      - virtual_step: learned surrogate prediction (cheap experiment)
      - check_invariants: conservation constraints that must ALWAYS hold
      - to_virtual_state: the abstraction map A: X -> Z
    """

    algorithm_name: str = "base"

    @abstractmethod
    def to_virtual_state(self, physical: Any) -> VirtualState:
        ...

    @abstractmethod
    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        """Execute the explicit mechanism. Source of ground truth."""
        ...

    @abstractmethod
    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        """Cheap learned prediction. Never mutates physical reality."""
        ...

    @abstractmethod
    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        """Return list of violated invariant names (empty = conserved)."""
        ...

    @abstractmethod
    def feature_names(self) -> list[str]:
        ...

    def describe(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "virtualization": "physical -> virtual_state -> learned_prediction",
            "paradigm": "surrogate (observations->latent->prediction), not simulation",
        }
