"""Conservation constraints: quantities reality guarantees must be preserved.

context.txt trust requirement #2. A surrogate may be arbitrarily clever
about timing/behavior, but it must never predict -- and the virtual layer
must never execute -- an outcome that violates the algorithm's invariants.
Raft safety rules are therefore hardcoded, never learned.
"""

from __future__ import annotations

from typing import Any

from rlraft.core.raft_rules import decide_vote


class RaftInvariants:
    """Majority, one-vote-per-term, log-freshness. Mirrors rlraft safety."""

    @staticmethod
    def check_vote(
        current_term: int,
        voted_for: int | None,
        candidate_term: int,
        candidate_id: int,
        candidate_last_index: int,
        candidate_last_term: int,
        local_last_index: int,
        local_last_term: int,
    ) -> list[str]:
        d = decide_vote(
            current_term, voted_for, candidate_term, candidate_id,
            candidate_last_index, candidate_last_term,
            local_last_index, local_last_term,
        )
        violations: list[str] = []
        # Conservation: a granted vote must satisfy all three Raft constraints.
        if d.granted:
            if candidate_term < current_term:
                violations.append("stale_term_granted")
            if voted_for not in (None, candidate_id):
                violations.append("double_vote_granted")
            if candidate_last_term < local_last_term or (
                candidate_last_term == local_last_term
                and candidate_last_index < local_last_index
            ):
                violations.append("stale_log_granted")
        return violations

    @staticmethod
    def check_election_outcome(
        votes_by_candidate: dict[int, int],
        winner: int | None,
        cluster_size: int,
        success: bool,
    ) -> list[str]:
        violations: list[str] = []
        majority = cluster_size // 2 + 1
        if success:
            if winner is None:
                violations.append("success_without_winner")
            elif votes_by_candidate.get(winner, 0) < majority:
                violations.append("winner_without_majority")
        # vote conservation: no candidate can hold more votes than nodes
        for cand, votes in votes_by_candidate.items():
            if votes > cluster_size:
                violations.append(f"vote_overflow:{cand}")
            if votes < 0:
                violations.append(f"negative_votes:{cand}")
        return violations

    @staticmethod
    def check_prediction(outcome: dict[str, Any], cluster_size: int) -> list[str]:
        """Conservation check on surrogate-predicted outcomes."""
        violations: list[str] = []
        p_success = float(outcome.get("success_prob", 0.0))
        if not 0.0 <= p_success <= 1.0:
            violations.append("probability_out_of_range:success_prob")
        p_split = float(outcome.get("split_prob", 0.0))
        if not 0.0 <= p_split <= 1.0:
            violations.append("probability_out_of_range:split_prob")
        t = float(outcome.get("election_time_ms", 0.0))
        if t < 0:
            violations.append("negative_time")
        if t > 30_000:
            violations.append("unphysical_time")
        return violations


class BackoffInvariants:
    """Retry backoff: delays non-negative, bounded, monotone in failures."""

    MAX_DELAY_MS = 30_000.0

    @staticmethod
    def check_prediction(outcome: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        d = float(outcome.get("delay_ms", 0.0))
        if d < 0:
            violations.append("negative_delay")
        if d > BackoffInvariants.MAX_DELAY_MS:
            violations.append("unbounded_delay")
        return violations

    @staticmethod
    def check_monotone(delays: list[float]) -> list[str]:
        for i in range(1, len(delays)):
            if delays[i] < delays[i - 1] - 1e-9:
                return ["nonmonotone_backoff"]
        return []


class GossipInvariants:
    """Gossip dissemination: coverage in [0,1], rounds non-negative."""

    @staticmethod
    def check_prediction(outcome: dict[str, Any], cluster_size: int) -> list[str]:
        violations: list[str] = []
        cov = float(outcome.get("coverage", 0.0))
        if not 0.0 <= cov <= 1.0:
            violations.append("coverage_out_of_range")
        r = float(outcome.get("rounds", 0.0))
        if r < 0:
            violations.append("negative_rounds")
        infected = float(outcome.get("infected", 0.0))
        if infected < 0 or infected > cluster_size:
            violations.append("infected_out_of_range")
        return violations


class TCPInvariants:
    """AIMD congestion control conservation laws.

    Hard guarantees the surrogate must respect: window bounds,
    throughput cannot exceed bottleneck capacity (packet conservation),
    probabilities and times in range.
    """

    @staticmethod
    def check_prediction(outcome: dict[str, Any], bottleneck_mbps: float) -> list[str]:
        violations: list[str] = []
        cwnd = float(outcome.get("cwnd_pkts", 1.0))
        if cwnd < 1.0 - 1e-9:
            violations.append("cwnd_below_one_mss")
        cap = float(outcome.get("cap_pkts", cwnd))
        if cwnd > cap + 1e-9:
            violations.append("cwnd_exceeds_cap")
        thr = float(outcome.get("throughput_mbps", 0.0))
        if thr < 0:
            violations.append("negative_throughput")
        if thr > bottleneck_mbps * (1.0 + 1e-9):
            violations.append("throughput_exceeds_bottleneck")
        loss = float(outcome.get("loss_rate", 0.0))
        if not 0.0 <= loss <= 1.0:
            violations.append("loss_out_of_range")
        if float(outcome.get("rtt_ms", 0.0)) < 0:
            violations.append("negative_rtt")
        return violations

    @staticmethod
    def check_trace(cwnds: list[float], cap: float) -> list[str]:
        for c in cwnds:
            if c < 1.0 - 1e-9:
                return ["trace_cwnd_below_one_mss"]
            if c > cap + 1e-9:
                return ["trace_cwnd_exceeds_cap"]
        return []


class RoutingInvariants:
    """Overlay routing: latency bounded below by propagation, rates in range,
    paths loop-free and connected."""

    @staticmethod
    def check_prediction(
        outcome: dict[str, Any], prop_lower_bound_ms: float, path: list[int]
    ) -> list[str]:
        violations: list[str] = []
        lat = float(outcome.get("mean_latency_ms", 0.0))
        if lat < prop_lower_bound_ms - 1e-9:
            violations.append("latency_below_propagation_bound")
        if lat < 0:
            violations.append("negative_latency")
        rate = float(outcome.get("delivery_rate", 0.0))
        if not 0.0 <= rate <= 1.0:
            violations.append("delivery_out_of_range")
        if len(set(path)) != len(path):
            violations.append("path_has_loop")
        return violations

    @staticmethod
    def check_path_connected(path: list[int], edges: set[tuple[int, int]]) -> list[str]:
        undirected = edges | {(b, a) for a, b in edges}
        for a, b in zip(path, path[1:]):
            if (a, b) not in undirected:
                return [f"path_disconnected:{a}->{b}"]
        return []


class PagingInvariants:
    """Demand paging: resident set bounded by frames, faults are misses,
    Belady MIN is a lower bound on any realizable policy."""

    @staticmethod
    def check_outcome(
        faults: int, refs: int, max_resident: int, frames: int,
        min_faults: int | None = None,
    ) -> list[str]:
        violations: list[str] = []
        if faults < 0:
            violations.append("negative_faults")
        if faults > refs:
            violations.append("faults_exceed_refs")
        if max_resident > frames:
            violations.append("resident_exceeds_frames")
        if min_faults is not None and faults < min_faults:
            violations.append("beats_belady_impossible")
        return violations

    @staticmethod
    def check_prediction(fault_rate: float) -> list[str]:
        if not 0.0 <= fault_rate <= 1.0:
            return ["fault_rate_out_of_range"]
        return []


class SchedInvariants:
    """Scheduling: waits non-negative, server never exceeds unit rate,
    work conservation (idle only when backlog empty)."""

    @staticmethod
    def check_prediction(outcome: dict[str, Any]) -> list[str]:
        violations: list[str] = []
        if float(outcome.get("mean_wait_ms", 0.0)) < 0:
            violations.append("negative_wait")
        if float(outcome.get("mean_flow_ms", 0.0)) < 0:
            violations.append("negative_flow")
        util = float(outcome.get("utilization", 0.0))
        if not 0.0 <= util <= 1.0 + 1e-9:
            violations.append("utilization_out_of_range")
        return violations
