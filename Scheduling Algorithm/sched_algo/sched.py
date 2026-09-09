"""CPU/job scheduling virtualization.

Grounded in context.txt's computer-environment structure: "processes"
(~497) as hidden structure, "OS/hardware machinery maintains the
relationship" (~1007), and the general systems-virtualization program
(~1143-1166: memory, networks, storage, machines, ...).

Physical (P): single-server queue with stochastic arrivals (exponential
interarrivals) and bursts; policies FCFS / SJF-exact (burst knowledge
assumed -- reported as an oracle bound, never a deployable claim) /
round-robin with quantum q (preemptive, needs no burst knowledge).
Virtual (V): ridge-linear mean-waiting model fit from trajectories on
[utilization, mean_burst, quantum_norm, bias], used to rank disciplines.
/// Conservation: waits/flows non-negative, utilization in [0,1],
work conservation (idle only when backlog empty -- audited in-sim).
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
from rlraft.virtual.invariants import SchedInvariants
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker

DISCIPLINES = ["fcfs", "sjf", "rr_q1", "rr_q4"]


def generate_jobs(
    n: int, arrival_rate: float, mean_burst_ms: float, rng: random.Random,
    burst_cv: float = 0.5,
) -> list[tuple[float, float]]:
    """(arrival_ms, burst_ms) stream. arrival_rate is jobs per ms, so
    utilization rho = arrival_rate * mean_burst_ms. Bursts are
    mean * (1 + burst_cv * (2u - 1)), floored -- burst_cv = 0.5
    reproduces the legacy uniform[0.5, 1.5] stream exactly; higher values
    probe variability-driven queueing (Pollaczek-Khinchine regime)."""
    jobs = []
    t = 0.0
    for _ in range(n):
        t += rng.expovariate(arrival_rate)
        burst = mean_burst_ms * max(1.0 + burst_cv * (2.0 * rng.random() - 1.0), 0.05)
        jobs.append((t, burst))
    return jobs


def simulate_sched(
    jobs: list[tuple[float, float]], discipline: str, rng: random.Random,
) -> dict[str, float]:
    """Explicit single-server mechanism with work-conservation auditing.

    Waiting is measured exactly as flow - burst (total time in queue,
    including time re-queued between RR quanta). SJF assumes exact burst
    knowledge and is therefore reported as an oracle bound, not a
    deployable claim.
    """
    import heapq

    _ = rng  # mechanism is deterministic given the job stream
    bursts = {i: b for i, (_, b) in enumerate(jobs)}

    def quantum() -> float:
        if discipline == "rr_q1":
            return 1.0
        if discipline == "rr_q4":
            return 4.0
        return float("inf")  # fcfs / sjf run to completion

    def key(idx: int) -> float:
        if discipline == "sjf":
            return bursts[idx]
        return jobs[idx][0]  # fcfs / rr: FIFO tie-break by arrival

    order = sorted(range(len(jobs)), key=lambda i: jobs[i][0])
    ready_q: list[tuple[float, float, int, float]] = []  # (key, arrival, idx, remaining)
    i = 0
    t = 0.0
    flows: list[float] = []
    idle_time = 0.0
    while i < len(jobs) or ready_q:
        if not ready_q:
            t = max(t, jobs[order[i]][0])
        while i < len(jobs) and jobs[order[i]][0] <= t:
            j = order[i]
            heapq.heappush(ready_q, (key(j), jobs[j][0], j, bursts[j]))
            i += 1
        if not ready_q:
            continue
        _, _, idx, rem = heapq.heappop(ready_q)
        start = max(t, jobs[idx][0])
        idle_time += max(0.0, start - t)
        run = min(rem, quantum())
        t = start + run
        rem -= run
        while i < len(jobs) and jobs[order[i]][0] <= t:
            j = order[i]
            heapq.heappush(ready_q, (key(j), jobs[j][0], j, bursts[j]))
            i += 1
        if rem > 1e-9:
            heapq.heappush(ready_q, (key(idx), jobs[idx][0], idx, rem))
        else:
            flows.append(t - jobs[idx][0])
    busy_time = sum(bursts.values())
    span = max(t, 1e-9)
    waits = _exact_waits(jobs, discipline)
    return {
        "mean_wait_ms": sum(waits) / max(len(waits), 1),
        "mean_flow_ms": sum(flows) / max(len(flows), 1),
        "utilization": min(busy_time / span, 1.0),
        "idle_time_ms": idle_time,
        "jobs": float(len(jobs)),
    }


def _exact_waits(jobs: list[tuple[float, float]], discipline: str) -> list[float]:
    """Per-job waiting (flow - burst) via a second instrumented pass."""
    import heapq

    bursts = {i: b for i, (_, b) in enumerate(jobs)}

    def quantum() -> float:
        return 1.0 if discipline == "rr_q1" else 4.0 if discipline == "rr_q4" else float("inf")

    def key(idx: int) -> float:
        return bursts[idx] if discipline == "sjf" else jobs[idx][0]

    order = sorted(range(len(jobs)), key=lambda i: jobs[i][0])
    ready_q: list[tuple[float, float, int, float]] = []
    i = 0
    t = 0.0
    completion: dict[int, float] = {}
    while i < len(jobs) or ready_q:
        if not ready_q:
            t = max(t, jobs[order[i]][0])
        while i < len(jobs) and jobs[order[i]][0] <= t:
            j = order[i]
            heapq.heappush(ready_q, (key(j), jobs[j][0], j, bursts[j]))
            i += 1
        if not ready_q:
            continue
        _, _, idx, rem = heapq.heappop(ready_q)
        t = max(t, jobs[idx][0]) + min(rem, quantum())
        rem -= min(rem, quantum())
        while i < len(jobs) and jobs[order[i]][0] <= t:
            j = order[i]
            heapq.heappush(ready_q, (key(j), jobs[j][0], j, bursts[j]))
            i += 1
        if rem > 1e-9:
            heapq.heappush(ready_q, (key(idx), jobs[idx][0], idx, rem))
        else:
            completion[idx] = t
    return [completion[k] - jobs[k][0] - bursts[k] for k in sorted(completion)]


def sched_features(utilization: float, mean_burst_ms: float, quantum_ms: float,
                   burst_info: float = 0.0, burst_cv: float = 0.5) -> list[float]:
    """burst_info=1 marks disciplines that use exact burst knowledge (SJF).
    Without it FCFS and SJF are feature-identical and no model can separate
    them -- the feature, not the fitter, would be at fault. burst_cv and
    the util*cv interaction encode variability-driven queueing
    (Pollaczek-Khinchine: waits grow with 1 + CV^2 at fixed load)."""
    qnorm = 0.0 if quantum_ms == float("inf") else min(quantum_ms / 8.0, 1.5)
    u = min(utilization, 1.5)
    return [u, min(mean_burst_ms / 20.0, 2.0), qnorm, burst_info,
            min(burst_cv, 2.0), min(u * burst_cv, 3.0), 1.0]


SCHED_FEATURE_NAMES = ["utilization", "burst_norm", "quantum_norm", "burst_info",
                       "burst_cv", "util_x_cv", "bias"]

_QUANTUM_OF = {"fcfs": float("inf"), "sjf": float("inf"), "rr_q1": 1.0, "rr_q4": 4.0}
_BURST_INFO_OF = {"fcfs": 0.0, "sjf": 1.0, "rr_q1": 0.0, "rr_q4": 0.0}


class SchedWaitModel:
    """Ridge-linear mean-waiting surrogate fit from trajectories."""

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.weights: list[float] = []
        self.residual_std = 0.0

    def fit(self, rows: list[tuple[list[float], float]]) -> "SchedWaitModel":
        import numpy as np

        X = np.asarray([r[0] for r in rows], dtype=float)
        y = np.asarray([r[1] for r in rows], dtype=float)
        d = X.shape[1]
        w = np.linalg.solve(X.T @ X + self.reg * np.eye(d), X.T @ y)
        self.weights = w.tolist()
        resid = y - X @ w
        self.residual_std = float(np.sqrt(max(resid @ resid / max(len(y), 1), 0.0)))
        return self

    def predict(self, feats: list[float]) -> float:
        return max(sum(w * f for w, f in zip(self.weights, feats)), 0.0)


def fit_ensemble(
    rows: list[tuple[list[float], float, dict[str, float]]],
    members: int = 5,
    epochs: int = 250,
    seed: int = 7,
    wait_scale: float = 50.0,
) -> Any:
    """Deep-ensemble surrogate on normalized waits (w/wait_scale).

    Nonlinear counterpart to SchedWaitModel (kept as ablation baseline).
    """
    from rlraft.virtual.surrogate import EnsembleRegressor

    ens = EnsembleRegressor(input_dim=len(SCHED_FEATURE_NAMES), output_dim=1,
                            members=members, hidden_dim=32)
    ens.train([(f, [w / wait_scale]) for f, w, _ in rows],
              epochs=epochs, seed=seed)
    return ens


def ensemble_predict(ensemble: Any, feats: list[float],
                     wait_scale: float = 50.0) -> tuple[float, float]:
    """(mean_wait_ms, std_ms)."""
    mean, std = ensemble.predict(feats)
    return max(mean[0], 0.0) * wait_scale, max(std[0], 0.0) * wait_scale


def collect_sched_trajectories(
    loads: list[dict[str, float]] | None = None,
    jobs_per_setting: int = 120, seed: int = 0,
) -> list[tuple[list[float], float, dict[str, float]]]:
    loads = loads or [
        {"arrival_rate": 0.02, "mean_burst_ms": 10.0, "burst_cv": 0.5},   # rho = 0.2
        {"arrival_rate": 0.06, "mean_burst_ms": 10.0, "burst_cv": 0.5},   # rho = 0.6
        {"arrival_rate": 0.04, "mean_burst_ms": 20.0, "burst_cv": 0.5},   # rho = 0.8
        {"arrival_rate": 0.05, "mean_burst_ms": 12.0, "burst_cv": 1.2},   # rho = 0.6, high variability
    ]
    rng = random.Random(seed)
    rows = []
    for load in loads:
        cv = load.get("burst_cv", 0.5)
        for disc in DISCIPLINES:
            waits = []
            for _ in range(4):
                jobs = generate_jobs(jobs_per_setting, load["arrival_rate"],
                                     load["mean_burst_ms"], rng, cv)
                waits.append(simulate_sched(jobs, disc, rng)["mean_wait_ms"])
            util = load["arrival_rate"] * load["mean_burst_ms"]
            feats = sched_features(util, load["mean_burst_ms"], _QUANTUM_OF[disc],
                                   _BURST_INFO_OF[disc], cv)
            rows.append((feats, sum(waits) / len(waits), dict(load)))
    return rows


@register_virtual_algorithm("scheduling")
class Scheduling(VirtualAlgorithm):
    """Queueing discipline virtualized: rank FCFS/SJF/RR by learned waits.

    Two surrogates: wait ensemble MLP (primary) + ridge-linear (ablation
    baseline). virtual_step prefers the ensemble when fitted.
    """

    def __init__(
        self,
        model: SchedWaitModel | None = None,
        ensemble: Any = None,
        ood: FeatureDistribution | None = None,
        gate: TrustGate | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.model = model
        self.ensemble = ensemble
        self.ood = ood or FeatureDistribution()
        self.gate = gate or TrustGate()
        self.calibrator = calibrator or Calibrator()
        self.tracker = UncertaintyTracker()

    def feature_names(self) -> list[str]:
        return list(SCHED_FEATURE_NAMES)

    def to_virtual_state(self, physical: Any) -> VirtualState:
        util = float(physical.get("arrival_rate", 0.03)) * float(physical.get("mean_burst_ms", 10.0))
        disc = str(physical.get("discipline", "fcfs"))
        cv = float(physical.get("burst_cv", 0.5))
        feats = sched_features(util, float(physical.get("mean_burst_ms", 10.0)),
                               _QUANTUM_OF.get(disc, float("inf")),
                               _BURST_INFO_OF.get(disc, 0.0), cv)
        return VirtualState(features=feats, feature_names=self.feature_names(),
                            metadata=dict(physical))

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        rng = rng or random.Random(0)
        disc = str(intervention.parameters.get("discipline", "fcfs"))
        n = int(intervention.parameters.get("jobs", 120))
        jobs = generate_jobs(n, float(physical.get("arrival_rate", 0.03)),
                             float(physical.get("mean_burst_ms", 10.0)), rng,
                             float(physical.get("burst_cv", 0.5)))
        out = simulate_sched(jobs, disc, rng)
        out["discipline"] = disc
        out["invariant_violations"] = SchedInvariants.check_prediction(out)
        return out

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        disc = str(intervention.parameters.get("discipline", "fcfs"))
        util = float(state.metadata.get("arrival_rate", 0.03)) * float(state.metadata.get("mean_burst_ms", 10.0))
        cv = float(state.metadata.get("burst_cv", 0.5))
        feats = sched_features(util, float(state.metadata.get("mean_burst_ms", 10.0)),
                               _QUANTUM_OF.get(disc, float("inf")),
                               _BURST_INFO_OF.get(disc, 0.0), cv)
        ood_score = self.ood.ood_score(feats)
        if self.ensemble is not None:
            wait, std = ensemble_predict(self.ensemble, feats)
            uncertainty = min(std / 20.0 + 0.1 * ood_score + 0.02, 1.0)
        elif self.model is not None and self.model.weights:
            wait = self.model.predict(feats)
            uncertainty = min(self.model.residual_std / 20.0 + 0.1 * ood_score + 0.02, 1.0)
        else:
            return Prediction(outcome={}, uncertainty=0.9, ood_score=ood_score, trusted=False)
        outcome = {"mean_wait_ms": wait, "mean_flow_ms": wait,
                   "utilization": min(util, 1.0)}
        violations = SchedInvariants.check_prediction(outcome)
        trusted, _, _ = self.gate.decide(
            ood_score, uncertainty, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome=outcome, uncertainty=uncertainty,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        return SchedInvariants.check_prediction(outcome)

    def sweep(self, physical: dict[str, Any]) -> list[dict[str, Any]]:
        state = self.to_virtual_state(physical)
        rows = []
        for disc in DISCIPLINES:
            pred = self.virtual_step(state, Intervention("set_discipline", {"discipline": disc}))
            rows.append({"discipline": disc,
                         **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in pred.outcome.items()},
                         "uncertainty": round(pred.uncertainty, 4),
                         "ood": round(pred.ood_score, 4), "trusted": pred.trusted})
        return rows

    def audit(self, load: dict[str, float], jobs: int = 120,
              repeats: int = 8, seed: int = 0) -> dict[str, Any]:
        """Lower-is-better waits mapped to higher-is-better scores for the
        shared ranking harness via negation. Ensemble primary, linear
        ablation, per-sample uncertainty validity."""
        from rlraft.virtual.eval import (
            bootstrap_ci, interval_coverage, ranking_consistency,
            uncertainty_validity, value_consistency,
        )

        rng = random.Random(seed)
        predicted: dict[str, float] = {}
        linear_pred: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        abs_errs, stds = [], []
        util = load["arrival_rate"] * load["mean_burst_ms"]
        cv = load.get("burst_cv", 0.5)
        for disc in DISCIPLINES:
            feats = sched_features(util, load["mean_burst_ms"], _QUANTUM_OF[disc],
                                   _BURST_INFO_OF[disc], cv)
            if self.ensemble is not None:
                wait, std = ensemble_predict(self.ensemble, feats)
                predicted[disc] = -wait
                stds.append(std)
            elif self.model is not None and self.model.weights:
                predicted[disc] = -self.model.predict(feats)
                stds.append(min(self.model.residual_std + 0.5, 50.0))
            else:
                predicted[disc] = 0.0
                stds.append(50.0)
            if self.model is not None and self.model.weights:
                linear_pred[disc] = -self.model.predict(feats)
            else:
                linear_pred[disc] = 0.0
            samples = []
            for _ in range(repeats):
                js = generate_jobs(jobs, load["arrival_rate"], load["mean_burst_ms"], rng, cv)
                samples.append(-simulate_sched(js, disc, rng)["mean_wait_ms"])
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[disc] = {"success_rate": sum(samples) / len(samples),
                              "success_lo": lo, "success_hi": hi}
            abs_errs.append(abs(predicted[disc] - empirics[disc]["success_rate"]))
        # per-sample uncertainty validity: 4 disciplines x repeats points
        ep_errs, ep_stds = [], []
        rng2 = random.Random(seed + 77)
        for disc in DISCIPLINES:
            s = stds[DISCIPLINES.index(disc)]
            for _ in range(repeats):
                js = generate_jobs(jobs, load["arrival_rate"], load["mean_burst_ms"], rng2, cv)
                sample = -simulate_sched(js, disc, rng2)["mean_wait_ms"]
                ep_errs.append(abs(predicted[disc] - sample))
                ep_stds.append(s)
        return {"value": value_consistency(predicted, empirics),
                "ranking": ranking_consistency(predicted, empirics),
                "load": load,
                "ablation_linear": {
                    "value": value_consistency(linear_pred, empirics),
                    "ranking": ranking_consistency(linear_pred, empirics),
                },
                "uncertainty": {
                    **uncertainty_validity(stds, abs_errs),
                    "interval_coverage_k2": interval_coverage(abs_errs, stds),
                    "per_episode": {
                        **uncertainty_validity(ep_stds, ep_errs),
                        "interval_coverage_k2": interval_coverage(ep_errs, ep_stds),
                        "n": len(ep_errs),
                    },
                }}
