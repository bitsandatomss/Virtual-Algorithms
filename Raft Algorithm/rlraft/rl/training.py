from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

from .guidance import AdvisorGuidance, get_guidance, guidance_to_dict
from rlraft.sim.sim import (
    FEATURE_NAMES,
    TIMEOUT_ARMS,
    NodeObservation,
    TabularQTimeoutPolicy,
    generate_conditions,
    observation_bucket,
    observation_quality_score,
    observations_from_conditions,
    simulate_election_round,
    simulate_failover,
)


@dataclass(slots=True)
class TrainingResult:
    episodes: int
    nodes: int
    success_rate: float
    split_vote_rate: float
    best_node_win_rate: float
    average_failover_ms: float
    policy_path: str


BUCKETS = [f"q{i}" for i in range(50)] + [f"q{i}:split" for i in range(50)] + ["default"]


def train_policy(
    episodes: int = 12000,
    nodes: int = 50,
    output_path: str = "runs/learned_policy.json",
    seed: int = 7,
    learning_rate: float = 0.16,
    advisor: str = "off",
    advisor_interval: int = 5000,
    require_llm: bool = False,
) -> TrainingResult:
    """Train shared tabular multi-agent Q learners through practice elections.

    All nodes execute the same observation-to-timeout policy, but every node
    contributes its own local transition after each election. This is parameter
    sharing across homogeneous Raft agents: scalable enough for 50-200 nodes,
    while still decentralized at execution time.
    """

    rng = random.Random(seed)
    arms = list(TIMEOUT_ARMS)
    q_values = {
        bucket: _initial_bucket_values(bucket, arms)
        for bucket in BUCKETS
    }
    visits = {
        bucket: {arm: 0 for arm in arms}
        for bucket in BUCKETS
    }
    if require_llm:
        raise RuntimeError("LLM reward-shaping advisors are disabled; use llm_mappo.")
    guidance = get_guidance(advisor=advisor)
    advisor_history = [guidance_to_dict(guidance)]
    rolling = _RollingMetrics()

    for episode in range(episodes):
        if episode > 0 and advisor_interval > 0 and episode % advisor_interval == 0:
            guidance = get_guidance(
                advisor=advisor,
                metrics=rolling.metrics(),
            )
            advisor_history.append({"episode": episode, **guidance_to_dict(guidance)})
            rolling = _RollingMetrics()

        epsilon = max(0.02, 0.45 * (1.0 - episode / max(episodes, 1)))
        recent_split = rng.random() < 0.25
        conditions = generate_conditions(nodes, rng)
        observations = observations_from_conditions(conditions, recent_split=recent_split)
        chosen_actions: dict[int, str] = {}
        fixed_timeouts: dict[int, float] = {}

        for node_id, observation in observations.items():
            bucket = observation_bucket(observation)
            action = _epsilon_greedy(q_values[bucket], epsilon, rng)
            low, high = TIMEOUT_ARMS[action]
            chosen_actions[node_id] = action
            fixed_timeouts[node_id] = rng.uniform(low, high)

        result = _simulate_with_fixed_timeouts(conditions, fixed_timeouts, rng, recent_split)
        rolling.add(result)
        global_reward = _global_reward(result, guidance)

        for node_id, observation in observations.items():
            bucket = observation_bucket(observation)
            action = chosen_actions[node_id]
            reward = global_reward + _local_reward(
                node_id=node_id,
                observation=observation,
                action=action,
                leader_id=result.leader_id,
                best_node_id=result.best_node_id,
                guidance=guidance,
            )
            visits[bucket][action] += 1
            alpha = learning_rate / (1.0 + visits[bucket][action] ** 0.35)
            q_values[bucket][action] += alpha * (reward - q_values[bucket][action])

    eval_metrics = evaluate_policy(q_values, nodes=nodes, seed=seed + 100_000, episodes=1000)
    artifact = {
        "policy_type": "tabular_marl_timeout_q_policy",
        "algorithm": "independent-q-learning-with-parameter-sharing",
        "episodes": episodes,
        "nodes": nodes,
        "seed": seed,
        "features": FEATURE_NAMES,
        "state_buckets": BUCKETS,
        "arms": TIMEOUT_ARMS,
        "q_values": q_values,
        "greedy_policy": {
            **_greedy_policy(q_values)
        },
        "advisor_default": "off",
        "advisor_history": advisor_history,
        "evaluation_metrics": eval_metrics,
        "research_note": (
            "Nodes observe local RTT/loss/log freshness only. Training aggregates "
            "experience from many agents; execution is decentralized."
        ),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return TrainingResult(
        episodes=episodes,
        nodes=nodes,
        success_rate=eval_metrics["success_rate"],
        split_vote_rate=eval_metrics["split_vote_rate"],
        best_node_win_rate=eval_metrics["best_node_win_rate"],
        average_failover_ms=eval_metrics["average_failover_ms"],
        policy_path=str(path),
    )


def evaluate_policy(
    q_values: dict[str, dict[str, float]],
    nodes: int = 50,
    seed: int = 17,
    episodes: int = 1000,
) -> dict[str, float]:
    rng = random.Random(seed)
    policy = TabularQTimeoutPolicy(q_values)
    successes = 0
    split_votes = 0
    best_wins = 0
    total_time = 0.0
    total_rounds = 0
    total_candidates = 0
    for _ in range(episodes):
        conditions = generate_conditions(nodes, rng)
        result = simulate_failover(conditions, policy, rng)
        successes += int(result.success)
        split_votes += result.split_votes
        best_wins += int(result.best_node_won)
        total_time += result.total_time_ms
        total_rounds += result.rounds
        total_candidates += result.candidates_started
    return {
        "success_rate": successes / episodes,
        "split_vote_rate": split_votes / episodes,
        "best_node_win_rate": best_wins / episodes,
        "average_failover_ms": total_time / episodes,
        "average_rounds": total_rounds / episodes,
        "average_candidates_started": total_candidates / episodes,
    }


def load_learned_policy(path: str) -> TabularQTimeoutPolicy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("policy_type") != "tabular_marl_timeout_q_policy":
        raise ValueError(f"unsupported learned policy artifact: {data.get('policy_type')}")
    return TabularQTimeoutPolicy(q_values=data["q_values"], arms=data.get("arms"))


def _simulate_with_fixed_timeouts(conditions, fixed_timeouts, rng, recent_split: bool = False):
    class FixedPolicy:
        def timeout_ms(self, observation: NodeObservation, _rng: random.Random) -> float:
            return fixed_timeouts[observation.node_id]

    return simulate_election_round(conditions, FixedPolicy(), rng, recent_split=recent_split)


def _epsilon_greedy(values: dict[str, float], epsilon: float, rng: random.Random) -> str:
    if rng.random() < epsilon:
        return rng.choice(list(values))
    return max(values, key=values.get)


def _global_reward(result, guidance: AdvisorGuidance) -> float:
    reward = 2.5 if result.success else -2.5
    reward -= result.election_time_ms / guidance.failover_scale_ms
    reward -= guidance.split_penalty if result.split_vote else 0.0
    reward -= max(0, result.candidates_started - 1) * guidance.candidate_penalty
    if result.success:
        reward += (
            guidance.best_leader_bonus
            if result.leader_id == result.best_node_id
            else -guidance.non_best_leader_penalty
        )
    return reward


def _local_reward(
    node_id: int,
    observation: NodeObservation,
    action: str,
    leader_id: int | None,
    best_node_id: int,
    guidance: AdvisorGuidance,
) -> float:
    score = observation_quality_score(observation)
    early = action in {"very_short", "short"}
    late = action in {"long", "very_long"}
    reward = 0.0

    if node_id == leader_id:
        reward += 1.8 if node_id == best_node_id else -1.4
    else:
        if observation.recent_split and early:
            reward -= guidance.split_early_penalty
        if observation.recent_split and late:
            reward += guidance.split_late_bonus
        if score >= 0.35 and late:
            reward += 0.9
        if score >= guidance.early_quality_threshold and early:
            reward -= 1.1
        if score < guidance.elite_quality_threshold and early:
            reward += 0.6
        if score < 0.18 and late:
            reward -= 0.3
    return reward


def _greedy_policy(q_values: dict[str, dict[str, float]]) -> dict[str, str]:
    return {
        bucket: max(q_values[bucket], key=q_values[bucket].get)
        for bucket in BUCKETS
    }


class _RollingMetrics:
    def __init__(self) -> None:
        self.count = 0
        self.successes = 0
        self.splits = 0
        self.best = 0
        self.time = 0.0
        self.candidates = 0

    def add(self, result) -> None:
        self.count += 1
        self.successes += int(result.success)
        self.splits += int(result.split_vote)
        self.best += int(result.leader_id == result.best_node_id)
        self.time += result.election_time_ms
        self.candidates += result.candidates_started

    def metrics(self) -> dict[str, float]:
        if self.count == 0:
            return {}
        return {
            "success_rate": self.successes / self.count,
            "split_vote_rate": self.splits / self.count,
            "best_node_win_rate": self.best / self.count,
            "average_election_time_ms": self.time / self.count,
            "average_candidates_started": self.candidates / self.count,
        }


def _initial_bucket_values(bucket: str, arms: list[str]) -> dict[str, float]:
    quality = bucket.split(":")[0]
    if quality.startswith("q"):
        index = int(quality[1:])
    else:
        index = 5
    if index <= 2:
        preferred = "short"
    elif index <= 10:
        preferred = "medium"
    elif index <= 25:
        preferred = "long"
    else:
        preferred = "very_long"
    if bucket.endswith(":split"):
        preferred = "long" if index <= 3 else "very_long"
    preferred_index = arms.index(preferred)
    return {
        arm: -0.08 * abs(index - preferred_index)
        for index, arm in enumerate(arms)
    }
