from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoteDecision:
    term: int
    granted: bool
    voted_for: int | None
    step_down: bool = False
    reason: str = ""


def is_candidate_log_up_to_date(
    candidate_last_index: int,
    candidate_last_term: int,
    local_last_index: int,
    local_last_term: int,
) -> bool:
    if candidate_last_term != local_last_term:
        return candidate_last_term > local_last_term
    return candidate_last_index >= local_last_index


def decide_vote(
    current_term: int,
    voted_for: int | None,
    candidate_term: int,
    candidate_id: int,
    candidate_last_index: int,
    candidate_last_term: int,
    local_last_index: int,
    local_last_term: int,
) -> VoteDecision:
    if candidate_term < current_term:
        return VoteDecision(current_term, False, voted_for, reason="stale_term")

    new_term = max(current_term, candidate_term)
    new_voted_for = voted_for
    step_down = candidate_term > current_term

    if step_down:
        new_voted_for = None

    if new_voted_for not in (None, candidate_id):
        return VoteDecision(new_term, False, new_voted_for, step_down, "already_voted")

    if not is_candidate_log_up_to_date(
        candidate_last_index,
        candidate_last_term,
        local_last_index,
        local_last_term,
    ):
        return VoteDecision(new_term, False, new_voted_for, step_down, "stale_log")

    return VoteDecision(new_term, True, candidate_id, step_down, "granted")
