"""Concrete virtual algorithms (plural).

Each entry virtualizes one algorithmic behavior while holding its
conservation invariants fixed:

  1. RaftElection -- the RL-Raft result reframed: physical Raft
     election vs learned timeout virtualization. Safety rules never learned.
  2. Backoff -- binary exponential backoff vs learned delay table.
  3. Gossip -- fixed-fanout gossip vs learned fanout.

Together they demonstrate the general pattern:
  physical reality -> virtualization -> computation,
where the virtual object (state + ops + invariants) is what the agent
programs against.
"""

from __future__ import annotations

import random
from typing import Any

from rlraft.virtual.base import (
    Intervention,
    Prediction,
    VirtualAlgorithm,
    VirtualState,
    register_virtual_algorithm,
)
from rlraft.virtual.invariants import BackoffInvariants, GossipInvariants, RaftInvariants
from rlraft.virtual.surrogate import cluster_features
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker


@register_virtual_algorithm("raft_election")
class RaftElection(VirtualAlgorithm):
    """RL-Raft as a virtual algorithm.

    physical_step: deterministic simulator with an explicit timeout policy.
    virtual_step: DynamicsSurrogate prediction of the failover outcome for
      a candidate timeout-arm intervention (no simulation executed).
    Invariants: Raft vote conservation + predicted-probability conservation.
    """

    ARMS = ["very_short", "short", "medium", "long", "very_long"]

    def __init__(
        self,
        surrogate=None,
        ood: FeatureDistribution | None = None,
        gate: TrustGate | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.ood = ood or FeatureDistribution()
        self.gate = gate or TrustGate()
        self.calibrator = calibrator or Calibrator()
        self.tracker = UncertaintyTracker()

    def feature_names(self) -> list[str]:
        from rlraft.virtual.surrogate import FEATURE_NAMES

        return list(FEATURE_NAMES)

    def to_virtual_state(self, physical: dict[str, Any]) -> VirtualState:
        arm_index = int(physical.get("arm_index", 2))
        feats = cluster_features(
            int(physical.get("nodes", 50)),
            float(physical.get("mean_rtt_ms", 120.0)),
            float(physical.get("mean_loss", 0.02)),
            float(physical.get("mean_log_gap", 0.2)),
            bool(physical.get("recent_split", False)),
            arm_index,
        )
        return VirtualState(features=feats, feature_names=self.feature_names(), metadata=dict(physical))

    def physical_step(self, physical: dict[str, Any], intervention: Intervention, rng: Any) -> dict[str, Any]:
        from rlraft.sim.sim import TIMEOUT_ARMS, generate_conditions, simulate_election_round, simulate_failover

        nodes = int(physical.get("nodes", 10))
        arm = str(intervention.parameters.get("arm", "medium"))
        low, high = TIMEOUT_ARMS.get(arm, TIMEOUT_ARMS["medium"])

        class _P:
            def timeout_ms(self, obs, r):  # noqa: ANN001
                return r.uniform(low, high)

        rng = rng or random.Random(0)
        conds = generate_conditions(nodes, rng)
        res = simulate_failover(conds, _P(), rng)
        # vote-conservation audit on a single fresh round (FailoverResult
        # aggregates rounds, so it carries no per-candidate vote table)
        audit = simulate_election_round(generate_conditions(nodes, rng), _P(), rng)
        violations = RaftInvariants.check_election_outcome(
            dict(audit.votes_by_candidate), audit.leader_id, nodes, audit.success,
        )
        return {
            "success": res.success,
            "leader_id": res.leader_id,
            "best_node_id": res.best_node_id,
            "total_time_ms": res.total_time_ms,
            "rounds": res.rounds,
            "split_votes": res.split_votes,
            "invariant_violations": violations,
        }

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        arm = str(intervention.parameters.get("arm", "medium"))
        arm_index = self.ARMS.index(arm) if arm in self.ARMS else 2
        feats = list(state.features)
        feats[-1] = arm_index / (len(self.ARMS) - 1)
        ood_score = self.ood.ood_score(feats)
        if self.surrogate is None:
            return Prediction(outcome={}, uncertainty=0.9, ood_score=ood_score, trusted=False)
        raw = self.surrogate.predict(feats)
        horizon = int(intervention.parameters.get("horizon", 1))
        uncertainty = self.tracker.propagate(float(raw["uncertainty"]) + 0.1 * ood_score, horizon)
        violations = RaftInvariants.check_prediction(
            {"success_prob": raw["success_prob"], "split_prob": min(raw["split_prob"], 1.0),
             "election_time_ms": raw["election_time_ms"]},
            int(state.metadata.get("nodes", 50)),
        )
        trusted, reasons, _m = self.gate.decide(
            ood_score, uncertainty, self.calibrator.ece(), violations, is_intervention=True,
        )
        outcome = {
            "success_prob": raw["success_prob"],
            "split_prob": raw["split_prob"],
            "election_time_ms": raw["election_time_ms"],
        }
        return Prediction(outcome=outcome, uncertainty=uncertainty, ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        nodes = int(physical.get("nodes", 50)) if isinstance(physical, dict) else 50
        return RaftInvariants.check_prediction(outcome, nodes)


@register_virtual_algorithm("backoff")
class Backoff(VirtualAlgorithm):
    """Virtualized retry backoff.

    Physical: binary exponential backoff delay = base * 2**failures + jitter,
    capped. Virtual: learned per-failure-bucket delay table fitted from
    observed (failures -> delay, success) trajectories.
    Invariant: delays non-negative, bounded, monotone non-decreasing.
    """

    def __init__(self, base_ms: float = 100.0, cap_ms: float = 8000.0) -> None:
        self.base_ms = base_ms
        self.cap_ms = cap_ms
        self.table: dict[int, float] = {}  # learned virtual delays
        self.ood = FeatureDistribution()
        self.gate = TrustGate(ood_budget=2.0, uncertainty_budget=0.6)
        self.calibrator = Calibrator()

    def feature_names(self) -> list[str]:
        return ["failures_norm", "base_norm"]

    def to_virtual_state(self, physical: Any) -> VirtualState:
        f = int(physical.get("failures", 0)) if isinstance(physical, dict) else 0
        return VirtualState(
            features=[min(f / 8.0, 1.5), min(self.base_ms / 1000.0, 2.0)],
            feature_names=self.feature_names(),
            metadata={"failures": f},
        )

    def physical_delay(self, failures: int, rng: Any) -> float:
        rng = rng or random.Random(0)
        return min(self.base_ms * (2.0 ** failures) + rng.uniform(0, self.base_ms * 0.2), self.cap_ms)

    def fit(self, rows: list[tuple[int, float]]) -> None:
        """Learn virtual delay table: median observed delay per failure bucket."""
        buckets: dict[int, list[float]] = {}
        for f, d in rows:
            buckets.setdefault(min(f, 8), []).append(d)
        for f, ds in buckets.items():
            ds.sort()
            self.table[f] = ds[len(ds) // 2]
        # enforce monotone invariant on the learned object itself
        ordered = [self.table.get(f, self.physical_delay(f, random.Random(0))) for f in range(9)]
        for i in range(1, len(ordered)):
            ordered[i] = max(ordered[i], ordered[i - 1])
        for f in range(9):
            self.table[f] = ordered[f]
        # in-sample calibration: normalized predicted vs empirical delay.
        # (Passing a real ECE here instead of a hardcoded 0.0 is what makes
        # the trust gate meaningful for this algo.)
        for f, d in rows:
            self.calibrator.add(
                min(self.table[min(f, 8)] / self.cap_ms, 1.0),
                min(d / self.cap_ms, 1.0),
            )

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        f = int(intervention.parameters.get("failures", 0))
        d = self.physical_delay(f, rng)
        return {"delay_ms": d, "failures": f}

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        f = int(intervention.parameters.get("failures", state.metadata.get("failures", 0)))
        ood_score = 0.0 if f <= 8 else 2.5
        delay = self.table.get(min(f, 8), self.physical_delay(min(f, 8), random.Random(0)))
        violations = BackoffInvariants.check_prediction({"delay_ms": delay})
        trusted, _reasons, _m = self.gate.decide(
            ood_score, 0.1, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome={"delay_ms": delay}, uncertainty=0.1 + ood_score * 0.2,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        return BackoffInvariants.check_prediction(outcome)

    def audit(self, seed: int = 0, samples_per_bucket: int = 20) -> dict[str, Any]:
        """Predictive audit (no decision semantics: failures are observed,
        not chosen). Reports normalized-delay MAE + CI coverage, OOD
        refusal rate for f in 9..14, and monotonicity. Maturity is capped
        at INTERPOLATIVE with an explicit reason -- a lookup table cannot
        be counterfactual, and the report says so."""
        from rlraft.virtual.eval import bootstrap_ci, ranking_consistency, value_consistency

        rng = random.Random(seed)
        predicted: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        for f in range(9):
            key = f"f{f}"
            pred = self.virtual_step(
                self.to_virtual_state({"failures": f}),
                Intervention("observe", {"failures": f}),
            )
            predicted[key] = min(pred.outcome["delay_ms"] / self.cap_ms, 1.0)
            samples = [min(self.physical_delay(f, rng) / self.cap_ms, 1.0)
                       for _ in range(samples_per_bucket)]
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[key] = {"success_rate": sum(samples) / len(samples),
                             "success_lo": lo, "success_hi": hi}
        refused = sum(
            1 for f in range(9, 15)
            if not self.virtual_step(
                self.to_virtual_state({"failures": f}),
                Intervention("observe", {"failures": f}),
            ).trusted
        )
        return {"value": value_consistency(predicted, empirics),
                "ranking": ranking_consistency(predicted, empirics),
                "ood_refusal_rate": refused / 6.0,
                "maturity": "INTERPOLATIVE",
                "maturity_reason": "lookup table: predictive only, no decision semantics"}


@register_virtual_algorithm("gossip")
class Gossip(VirtualAlgorithm):
    """Virtualized gossip dissemination.

    Physical: push gossip, each infected node contacts `fanout` random peers
    per round with per-contact success prob `p`. Virtual: analytic surrogate
    predicting coverage/rounds, fitted `p` from trajectories.
    Invariants: coverage in [0,1], infected in [0, N].
    """

    def __init__(self, nodes: int = 50, p: float = 0.7) -> None:
        self.nodes = nodes
        self.p = p  # learned parameter of the virtual object
        self.ood = FeatureDistribution()
        self.gate = TrustGate(ood_budget=2.0, uncertainty_budget=0.6)
        self.calibrator = Calibrator()

    def feature_names(self) -> list[str]:
        return ["fanout_norm", "p_est", "nodes_norm"]

    def to_virtual_state(self, physical: Any) -> VirtualState:
        fanout = int(physical.get("fanout", 3)) if isinstance(physical, dict) else 3
        return VirtualState(
            features=[min(fanout / 8.0, 1.5), self.p, min(self.nodes / 200.0, 1.5)],
            feature_names=self.feature_names(),
            metadata={"fanout": fanout, "nodes": self.nodes},
        )

    def fit_p(self, observed_coverages: list[float], fanout: int, rounds: int) -> float:
        """Fit per-contact success p by grid search against observed coverage."""
        best_p, best_err = self.p, float("inf")
        for cand in [i / 100 for i in range(5, 100, 5)]:
            pred = self._analytic_coverage(self.nodes, fanout, cand, rounds)
            err = sum((pred - o) ** 2 for o in observed_coverages) / max(len(observed_coverages), 1)
            if err < best_err:
                best_err, best_p = err, cand
        self.p = best_p
        # in-sample calibration: analytic prediction vs observations, so the
        # gate's ECE check measures something real instead of a hardcoded 0.0
        for o in observed_coverages:
            self.calibrator.add(
                self._analytic_coverage(self.nodes, fanout, best_p, rounds), o)
        return best_p

    @staticmethod
    def _analytic_coverage(nodes: int, fanout: int, p: float, rounds: int) -> float:
        infected = 1.0
        for _ in range(rounds):
            susceptible = nodes - infected
            # each infected contacts `fanout` peers; expected new infections
            expected_new = infected * fanout * p * (susceptible / nodes)
            infected = min(nodes, infected + expected_new)
        return infected / nodes

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        rng = rng or random.Random(0)
        fanout = int(intervention.parameters.get("fanout", 3))
        rounds = int(intervention.parameters.get("rounds", 5))
        p = float(intervention.parameters.get("p", 0.7))
        infected_set = {0}
        for _ in range(rounds):
            new: set[int] = set()
            for node in list(infected_set):
                for _ in range(fanout):
                    peer = rng.randrange(self.nodes)
                    if peer not in infected_set and rng.random() < p:
                        new.add(peer)
            infected_set |= new
        coverage = len(infected_set) / self.nodes
        return {"coverage": coverage, "infected": float(len(infected_set)), "rounds": float(rounds)}

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        fanout = int(intervention.parameters.get("fanout", state.metadata.get("fanout", 3)))
        rounds = int(intervention.parameters.get("rounds", 5))
        ood_score = 0.0 if 1 <= fanout <= 8 and 1 <= rounds <= 12 else 2.5
        coverage = self._analytic_coverage(self.nodes, fanout, self.p, rounds)
        outcome = {"coverage": coverage, "infected": coverage * self.nodes, "rounds": float(rounds)}
        violations = GossipInvariants.check_prediction(outcome, self.nodes)
        trusted, _r, _m = self.gate.decide(
            ood_score, 0.12, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome=outcome, uncertainty=0.12 + ood_score * 0.2,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        return GossipInvariants.check_prediction(outcome, self.nodes)

    def audit(
        self,
        fanouts: list[int] | None = None,
        rounds: int = 5,
        p_true: float = 0.7,
        episodes_per_fanout: int = 12,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Fanout-selection audit: rank candidate fanouts by predicted
        coverage (higher is better) against simulated ground truth.
        Caller fits p first via fit_p; the audit reports whether the
        fitted analytic model preserves the decision ordering."""
        from rlraft.virtual.eval import bootstrap_ci, ranking_consistency, value_consistency

        fanouts = fanouts or [1, 2, 3, 4, 6]
        rng = random.Random(seed)
        predicted: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        for fan in fanouts:
            key = f"fan{fan}"
            pred = self.virtual_step(
                self.to_virtual_state({"fanout": fan}),
                Intervention("set_fanout", {"fanout": fan, "rounds": rounds}),
            )
            predicted[key] = pred.outcome["coverage"]
            samples = [
                self.physical_step(
                    {}, Intervention("spread", {"fanout": fan, "rounds": rounds,
                                               "p": p_true}), rng,
                )["coverage"]
                for _ in range(episodes_per_fanout)
            ]
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[key] = {"success_rate": sum(samples) / len(samples),
                             "success_lo": lo, "success_hi": hi}
        return {"value": value_consistency(predicted, empirics),
                "ranking": ranking_consistency(predicted, empirics),
                "fitted_p": self.p, "true_p": p_true, "rounds": rounds}
