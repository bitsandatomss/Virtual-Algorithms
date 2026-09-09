from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class AdvisorGuidance:
    source: str = "off"
    split_penalty: float = 3.5
    best_leader_bonus: float = 2.2
    non_best_leader_penalty: float = 0.8
    candidate_penalty: float = 0.04
    failover_scale_ms: float = 700.0
    early_quality_threshold: float = 0.14
    elite_quality_threshold: float = 0.12
    split_early_penalty: float = 0.9
    split_late_bonus: float = 0.5
    notes: str = "Legacy q-learning guidance."


def deterministic_guidance(metrics: dict[str, float] | None = None) -> AdvisorGuidance:
    guidance = AdvisorGuidance(source="deterministic")
    metrics = metrics or {}
    split_rate = metrics.get("split_vote_rate", 0.0)
    best_rate = metrics.get("best_node_win_rate", 0.0)
    if split_rate > 0.20:
        guidance.split_penalty = 5.2
        guidance.split_early_penalty = 1.3
        guidance.split_late_bonus = 0.8
        guidance.candidate_penalty = 0.07
        guidance.notes = "Increased split/candidate penalties."
    if best_rate < 0.55:
        guidance.best_leader_bonus = 3.8
        guidance.non_best_leader_penalty = 1.4
        guidance.early_quality_threshold = 0.12
        guidance.elite_quality_threshold = 0.10
        guidance.notes = "Increased best-leader pressure."
    return guidance


def get_guidance(advisor: str, metrics: dict[str, float] | None = None) -> AdvisorGuidance:
    if advisor == "off":
        return AdvisorGuidance(source="off", notes="Legacy q-learning guidance disabled.")
    if advisor == "deterministic":
        return deterministic_guidance(metrics)
    raise ValueError("LLM reward-shaping advisors are intentionally disabled; use llm_mappo instead.")


def guidance_to_dict(guidance: AdvisorGuidance) -> dict[str, Any]:
    return asdict(guidance)
