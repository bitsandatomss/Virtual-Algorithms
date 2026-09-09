from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import random
from typing import Protocol


FEATURE_NAMES = ["bias", "rtt_norm", "loss_rate", "log_gap_norm", "recent_split"]

TIMEOUT_ARMS: dict[str, tuple[float, float]] = {
    "very_short": (150.0, 230.0),
    "short": (280.0, 420.0),
    "medium": (520.0, 760.0),
    "long": (900.0, 1300.0),
    "very_long": (1450.0, 2100.0),
}


@dataclass(frozen=True, slots=True)
class NodeCondition:
    node_id: int
    quality_score: float
    mean_rtt_ms: float
    loss_rate: float
    log_gap: int = 0
    alive: bool = True


@dataclass(frozen=True, slots=True)
class NodeObservation:
    node_id: int
    mean_rtt_ms: float
    loss_rate: float
    log_gap: int
    recent_split: bool = False

    def features(self) -> list[float]:
        return [
            1.0,
            min(self.mean_rtt_ms / 350.0, 1.5),
            min(max(self.loss_rate, 0.0), 1.0),
            min(self.log_gap / 8.0, 1.0),
            1.0 if self.recent_split else 0.0,
        ]


@dataclass(frozen=True, slots=True)
class ElectionRoundResult:
    success: bool
    leader_id: int | None
    best_node_id: int
    election_time_ms: float
    split_vote: bool
    candidates_started: int
    votes_by_candidate: dict[int, int]
    reason: str


@dataclass(frozen=True, slots=True)
class FailoverResult:
    success: bool
    leader_id: int | None
    best_node_id: int
    total_time_ms: float
    rounds: int
    split_votes: int
    candidates_started: int
    best_node_won: bool
    reason: str


class TimeoutPolicy(Protocol):
    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        ...


class StaticRandomPolicy:
    def __init__(self, low_ms: float = 150.0, high_ms: float = 300.0):
        self.low_ms = low_ms
        self.high_ms = high_ms

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        return rng.uniform(self.low_ms, self.high_ms)


class DynatuneLikePolicy:
    """Simple measurement-based baseline inspired by Dynatune's stated inputs.

    This is not a reproduction of Dynatune; it is a transparent baseline that
    maps heartbeat-style RTT/loss observations into a timeout.
    """

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        rtt = min(observation.mean_rtt_ms / 350.0, 1.0)
        loss = min(observation.loss_rate / 0.25, 1.0)
        log = min(observation.log_gap / 8.0, 1.0)
        penalty = 0.62 * rtt + 0.28 * loss + 0.10 * log
        if observation.recent_split:
            penalty += 0.12
        return 160.0 + penalty * 950.0 + rng.uniform(0.0, 90.0)


class LearnedSoftmaxPolicy:
    def __init__(
        self,
        weights: list[list[float]],
        arms: dict[str, tuple[float, float]] | None = None,
        greedy: bool = True,
    ):
        self.weights = weights
        self.arms = arms or TIMEOUT_ARMS
        self.arm_names = list(self.arms)
        self.greedy = greedy

    def action_probabilities(self, observation: NodeObservation) -> list[float]:
        features = observation.features()
        logits = [
            sum(weight * value for weight, value in zip(row, features))
            for row in self.weights
        ]
        return _softmax(logits)

    def choose_action(self, observation: NodeObservation, rng: random.Random) -> str:
        probs = self.action_probabilities(observation)
        if self.greedy:
            return self.arm_names[max(range(len(probs)), key=probs.__getitem__)]
        draw = rng.random()
        cumulative = 0.0
        for arm, probability in zip(self.arm_names, probs):
            cumulative += probability
            if draw <= cumulative:
                return arm
        return self.arm_names[-1]

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        arm = self.choose_action(observation, rng)
        low, high = self.arms[arm]
        return rng.uniform(low, high)


class LearnedLinearTimeoutPolicy:
    def __init__(
        self,
        coefficients: dict[str, float],
        min_ms: float = 145.0,
        max_ms: float = 2400.0,
        jitter_ms: float = 45.0,
    ):
        self.coefficients = coefficients
        self.min_ms = min_ms
        self.max_ms = max_ms
        self.jitter_ms = jitter_ms

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        features = dict(zip(FEATURE_NAMES, observation.features()))
        value = self.min_ms
        for name, coefficient in self.coefficients.items():
            value += coefficient * features.get(name, 0.0)
        value += rng.uniform(0.0, self.jitter_ms)
        return min(max(value, self.min_ms), self.max_ms)


class TabularQTimeoutPolicy:
    def __init__(
        self,
        q_values: dict[str, dict[str, float]],
        arms: dict[str, tuple[float, float]] | None = None,
    ):
        self.q_values = q_values
        self.arms = arms or TIMEOUT_ARMS

    def choose_action(self, observation: NodeObservation) -> str:
        bucket = observation_bucket(observation)
        values = self.q_values.get(bucket) or self.q_values.get("default")
        if not values:
            return "medium"
        return max(values, key=values.get)

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        action = self.choose_action(observation)
        low, high = self.arms[action]
        return rng.uniform(float(low), float(high))


def generate_conditions(
    nodes: int,
    rng: random.Random,
    failure_rate: float = 0.0,
    log_lag_rate: float = 0.05,
) -> list[NodeCondition]:
    raw_scores = sorted(rng.betavariate(1.1, 2.8) for _ in range(nodes))
    rng.shuffle(raw_scores)
    conditions: list[NodeCondition] = []
    for node_id, score in enumerate(raw_scores):
        mean_rtt = 24.0 + score * 280.0 + rng.uniform(-8.0, 14.0)
        loss = min(max(0.003 + score * 0.20 + rng.uniform(-0.004, 0.015), 0.0), 0.45)
        log_gap = rng.choice([1, 2, 3, 4, 6]) if rng.random() < log_lag_rate else 0
        alive = rng.random() >= failure_rate
        conditions.append(
            NodeCondition(
                node_id=node_id,
                quality_score=score,
                mean_rtt_ms=max(8.0, mean_rtt),
                loss_rate=loss,
                log_gap=log_gap,
                alive=alive,
            )
        )
    if sum(1 for node in conditions if node.alive) < nodes // 2 + 1:
        conditions = [condition if idx < nodes // 2 + 1 else condition for idx, condition in enumerate(conditions)]
    return conditions


def observations_from_conditions(
    conditions: list[NodeCondition],
    recent_split: bool = False,
) -> dict[int, NodeObservation]:
    return {
        condition.node_id: NodeObservation(
            node_id=condition.node_id,
            mean_rtt_ms=condition.mean_rtt_ms,
            loss_rate=condition.loss_rate,
            log_gap=condition.log_gap,
            recent_split=recent_split,
        )
        for condition in conditions
    }


def observation_bucket(observation: NodeObservation) -> str:
    score = observation_quality_score(observation)
    bucket_index = min(49, max(0, int(score * 50)))
    quality = f"q{bucket_index}"
    return f"{quality}:split" if observation.recent_split else quality


def observation_quality_score(observation: NodeObservation) -> float:
    return min(
        0.62 * min(observation.mean_rtt_ms / 320.0, 1.5)
        + 0.28 * min(observation.loss_rate / 0.22, 1.5)
        + 0.10 * min(observation.log_gap / 6.0, 1.0),
        1.2,
    )


def best_connected_node(conditions: list[NodeCondition]) -> int:
    alive = [condition for condition in conditions if condition.alive]
    return min(
        alive,
        key=lambda c: (c.mean_rtt_ms + 850.0 * c.loss_rate + 35.0 * c.log_gap, c.node_id),
    ).node_id


def simulate_failover(
    conditions: list[NodeCondition],
    policy: TimeoutPolicy,
    rng: random.Random,
    max_rounds: int = 6,
    majority: int | None = None,
) -> FailoverResult:
    majority = majority or (len(conditions) // 2 + 1)
    recent_split = False
    total_time = 0.0
    split_votes = 0
    candidates_started = 0
    best = best_connected_node(conditions)

    for round_index in range(max_rounds):
        result = simulate_election_round(
            conditions=conditions,
            policy=policy,
            rng=rng,
            recent_split=recent_split,
            majority=majority,
        )
        total_time += result.election_time_ms
        candidates_started += result.candidates_started
        if result.success:
            return FailoverResult(
                success=True,
                leader_id=result.leader_id,
                best_node_id=best,
                total_time_ms=total_time,
                rounds=round_index + 1,
                split_votes=split_votes,
                candidates_started=candidates_started,
                best_node_won=result.leader_id == best,
                reason=result.reason,
            )
        split_votes += int(result.split_vote)
        recent_split = True
        total_time += 35.0 + rng.uniform(0.0, 50.0)

    return FailoverResult(
        success=False,
        leader_id=None,
        best_node_id=best,
        total_time_ms=total_time,
        rounds=max_rounds,
        split_votes=split_votes,
        candidates_started=candidates_started,
        best_node_won=False,
        reason="no_majority_after_retries",
    )


def simulate_election_round(
    conditions: list[NodeCondition],
    policy: TimeoutPolicy,
    rng: random.Random,
    recent_split: bool = False,
    majority: int | None = None,
) -> ElectionRoundResult:
    majority = majority or (len(conditions) // 2 + 1)
    observations = observations_from_conditions(conditions, recent_split=recent_split)
    by_id = {condition.node_id: condition for condition in conditions}
    alive = {condition.node_id for condition in conditions if condition.alive}
    if len(alive) < majority:
        return ElectionRoundResult(False, None, best_connected_node(conditions), 0.0, False, 0, {}, "minority_alive")

    timeouts = {
        node_id: policy.timeout_ms(observations[node_id], rng)
        for node_id in alive
    }
    first_timeout = min(timeouts.values())
    round_deadline = first_timeout + 1200.0
    events: list[tuple[float, int, str, int, int | None]] = []
    sequence = 0
    for node_id, timeout in timeouts.items():
        heapq.heappush(events, (timeout, sequence, "timeout", node_id, None))
        sequence += 1

    candidates: set[int] = set()
    voted_for: dict[int, int] = {}
    votes_by_candidate: dict[int, int] = {}
    winner: int | None = None
    win_time = round_deadline

    while events:
        now, _seq, event_type, actor, other = heapq.heappop(events)
        if now > round_deadline:
            break
        if actor not in alive:
            continue

        if event_type == "timeout":
            if actor in voted_for:
                continue
            candidates.add(actor)
            voted_for[actor] = actor
            votes_by_candidate[actor] = 1
            if majority == 1:
                winner = actor
                win_time = now
                break
            for voter in alive:
                if voter == actor:
                    continue
                if not _request_delivered(conditions, actor, voter, rng):
                    continue
                arrival = now + _one_way_delay_ms(conditions, actor, voter, rng)
                heapq.heappush(events, (arrival, sequence, "request_vote", voter, actor))
                sequence += 1

        elif event_type == "request_vote":
            candidate = other
            if candidate is None or candidate not in candidates or candidate not in alive:
                continue
            if actor in voted_for:
                continue
            if not _candidate_log_is_fresh(by_id[candidate], by_id[actor]):
                continue
            voted_for[actor] = candidate
            if not _request_delivered(conditions, actor, candidate, rng):
                continue
            arrival = now + _one_way_delay_ms(conditions, actor, candidate, rng)
            heapq.heappush(events, (arrival, sequence, "vote_response", candidate, actor))
            sequence += 1

        elif event_type == "vote_response":
            candidate = actor
            voter = other
            if voter is None or candidate not in candidates:
                continue
            votes_by_candidate[candidate] = votes_by_candidate.get(candidate, 1) + 1
            if votes_by_candidate[candidate] >= majority:
                winner = candidate
                win_time = now
                break

    best = best_connected_node(conditions)
    if winner is None:
        top_vote = max(votes_by_candidate.values(), default=0)
        return ElectionRoundResult(
            success=False,
            leader_id=None,
            best_node_id=best,
            election_time_ms=round_deadline,
            split_vote=len(candidates) > 1 and top_vote < majority,
            candidates_started=len(candidates),
            votes_by_candidate=votes_by_candidate,
            reason="split_vote_or_lost_responses",
        )

    return ElectionRoundResult(
        success=True,
        leader_id=winner,
        best_node_id=best,
        election_time_ms=win_time,
        split_vote=False,
        candidates_started=len(candidates),
        votes_by_candidate=votes_by_candidate,
        reason="majority_reached",
    )


def _one_way_delay_ms(
    conditions: list[NodeCondition],
    src: int,
    dst: int,
    rng: random.Random,
) -> float:
    a = conditions[src]
    b = conditions[dst]
    mean = 8.0 + 0.35 * a.mean_rtt_ms + 0.35 * b.mean_rtt_ms
    jitter = rng.lognormvariate(math.log(1.0), 0.22)
    return max(2.0, mean * jitter / 2.0)


def _request_delivered(
    conditions: list[NodeCondition],
    src: int,
    dst: int,
    rng: random.Random,
) -> bool:
    loss = min(0.80, conditions[src].loss_rate + conditions[dst].loss_rate)
    return rng.random() >= loss


def _candidate_log_is_fresh(candidate: NodeCondition, voter: NodeCondition) -> bool:
    return candidate.log_gap <= voter.log_gap


def _softmax(logits: list[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(min(30.0, value - top)) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]
