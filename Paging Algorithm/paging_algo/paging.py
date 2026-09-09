"""Demand-paging virtualization.

Directly grounded in context.txt: "physical memory -> virtual address
space"; "the physical RAM has pages, addresses, contention, caches, DMA"
(~1002-1005); "a process can manipulate a virtual address even though no
such physical object exists at the RAM level" (~1056).

Physical (P): reference string with locality (hot set + working-set
shifts), F frames, policies LRU / LFU / RANDOM, plus Belady MIN oracle
(computed with future knowledge -- lower-bound reference only, never an
intervention).
Virtual (V): (a) ridge-linear fault-rate models per policy fit from
workload trajectories, used to rank policies per workload; (b) a LEARNED
eviction policy: victim score = w_rec * recency_rank + w_freq * freq_rank
with weights grid-fit to minimize faults on training workloads, evaluated
held-out against LRU with the MIN gap reported.
/// Conservation: resident set <= frames, faults <= refs, no realizable
policy beats Belady MIN.
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
from rlraft.virtual.invariants import PagingInvariants
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker

POLICIES = ["lru", "lfu", "random"]


def generate_refs(
    pages: int, hot: int, length: int, shift_every: int, rng: random.Random,
) -> list[int]:
    """Locality workload: uniform over hot set, hot set shifts periodically."""
    refs = []
    base = 0
    for t in range(length):
        if t > 0 and t % shift_every == 0:
            base = rng.randrange(pages)
        if rng.random() < 0.85:
            refs.append((base + rng.randrange(hot)) % pages)
        else:
            refs.append(rng.randrange(pages))
    return refs


def simulate_paging(
    refs: list[int], frames: int, policy: str, rng: random.Random,
) -> dict[str, float]:
    """Explicit replacement mechanism with conservation auditing."""
    resident: dict[int, dict[str, float]] = {}  # page -> {last, freq}
    faults = 0
    max_resident = 0
    time = 0
    for p in refs:
        time += 1
        if p in resident:
            resident[p]["last"] = time
            resident[p]["freq"] += 1.0
            continue
        faults += 1
        if len(resident) >= frames:
            if policy == "lru":
                victim = min(resident, key=lambda q: resident[q]["last"])
            elif policy == "lfu":
                victim = min(resident, key=lambda q: (resident[q]["freq"], resident[q]["last"]))
            else:  # random
                victim = rng.choice(list(resident))
            del resident[victim]
        resident[p] = {"last": time, "freq": 1.0}
        max_resident = max(max_resident, len(resident))
    return {"faults": float(faults), "fault_rate": faults / max(len(refs), 1),
            "max_resident": float(max_resident), "refs": float(len(refs))}


def belady_min_faults(refs: list[int], frames: int) -> int:
    """Belady MIN oracle (future knowledge): lower bound for audit only."""
    resident: set[int] = set()
    faults = 0
    for i, p in enumerate(refs):
        if p in resident:
            continue
        faults += 1
        if len(resident) >= frames:
            # evict resident page with farthest next use
            future: dict[int, float] = {}
            for q in resident:
                nxt = float("inf")
                for j in range(i + 1, len(refs)):
                    if refs[j] == q:
                        nxt = j
                        break
                future[q] = nxt
            victim = max(future, key=lambda q: future[q])
            resident.remove(victim)
        resident.add(p)
    return faults


def reuse_stats(refs: list[int], cap: int = 800) -> tuple[float, float, float, float]:
    """Backward reuse distances (distinct refs since last use) on the first
    `cap` refs: (mean, p90, cold_frac, unique_ratio). The classic stack-depth
    quantity -- the information LRU/MIN implicitly exploit and the linear
    parametric features discard."""
    sample = refs[:cap]
    last: dict[int, int] = {}
    rds: list[float] = []
    cold = 0
    for i, p in enumerate(sample):
        if p not in last:
            cold += 1
            rds.append(float(len(set(sample[max(0, i - 64):i])) + 1))
        else:
            rds.append(float(len(set(sample[last[p] + 1:i]))))
        last[p] = i
    rds.sort()
    mean_rd = sum(rds) / max(len(rds), 1)
    p90 = rds[min(int(0.9 * len(rds)), len(rds) - 1)] if rds else 0.0
    return mean_rd, p90, cold / max(len(sample), 1), len(set(sample)) / max(len(sample), 1)


def _resolve_refs(pages: int, hot: int, length: int, shift_every: int,
                  refs: list[int] | None) -> list[int]:
    if refs is not None:
        return refs
    seed = (pages * 31 + hot * 17 + shift_every + length) % (2 ** 31)
    return generate_refs(pages, hot, length, shift_every, random.Random(seed))


def workload_features(pages: int, hot: int, shift_every: int, frames: int,
                      length: int, refs: list[int] | None = None) -> list[float]:
    """Parametric + reuse-distance features. Refs are resolved
    deterministically from params when not supplied, so features are a
    pure function of the workload description (no sampling lottery)."""
    resolved = _resolve_refs(pages, hot, length, shift_every, refs)
    mean_rd, p90_rd, cold_frac, unique_ratio = reuse_stats(resolved)
    return [hot / max(pages, 1), shift_every / max(length, 1),
            frames / max(pages, 1),
            min(mean_rd / max(frames, 1), 2.0),
            min(p90_rd / max(frames, 1), 3.0),
            cold_frac, unique_ratio, 1.0]


PAGING_FEATURE_NAMES = ["hot_frac", "shift_frac", "frames_frac", "mean_rd_frames",
                        "p90_rd_frames", "cold_frac", "unique_ratio", "bias"]


def policy_onehot(policy: str) -> list[float]:
    return [1.0 if policy == pol else 0.0 for pol in POLICIES]


def fit_fault_ensemble(
    rows: list[tuple[list[float], dict[str, float], list[int]]],
    members: int = 5,
    epochs: int = 250,
    seed: int = 7,
) -> Any:
    """One-hot MLP fault model: (feats + policy) -> fault_rate.

    Shares structure across policies instead of fitting isolated
    per-policy hyperplanes (the linear ablation). Uncertainty =
    ensemble std.
    """
    from rlraft.virtual.surrogate import EnsembleRegressor

    train = [(f + policy_onehot(pol), [rate])
             for f, rates, _, _ in rows for pol, rate in rates.items()]
    ens = EnsembleRegressor(input_dim=len(PAGING_FEATURE_NAMES) + len(POLICIES),
                            output_dim=1, members=members, hidden_dim=32)
    ens.train(train, epochs=epochs, seed=seed)
    return ens


class PolicyFaultModel:
    """Per-policy ridge-linear fault-rate models fit from trajectories."""

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.weights: dict[str, list[float]] = {}
        self.residual_std = 0.0

    def fit(self, rows: list[tuple[list[float], dict[str, float]]]) -> "PolicyFaultModel":
        import numpy as np

        X = np.asarray([r[0] for r in rows], dtype=float)
        d = X.shape[1]
        A = X.T @ X + self.reg * np.eye(d)
        resid_all = []
        for pol in POLICIES:
            y = np.asarray([r[1][pol] for r in rows], dtype=float)
            w = np.linalg.solve(A, X.T @ y)
            self.weights[pol] = w.tolist()
            resid_all.extend((y - X @ w).tolist())
        resid_all = np.asarray(resid_all)
        self.residual_std = float(np.sqrt(max(resid_all @ resid_all / max(len(resid_all), 1), 0.0)))
        return self

    def predict(self, feats: list[float]) -> dict[str, float]:
        import numpy as np

        x = np.asarray(feats, dtype=float)
        return {pol: float(np.clip(x @ np.asarray(w), 0.0, 1.0))
                for pol, w in self.weights.items()}


def collect_paging_trajectories(
    workloads: list[dict[str, int]] | None = None,
    seed: int = 0,
) -> list[tuple[list[float], dict[str, float], list[int]]]:
    workloads = workloads or [
        {"pages": 64, "hot": 8, "length": 1500, "shift_every": 500, "frames": 8},
        {"pages": 64, "hot": 16, "length": 1500, "shift_every": 300, "frames": 12},
        {"pages": 128, "hot": 10, "length": 2000, "shift_every": 700, "frames": 10},
        {"pages": 128, "hot": 32, "length": 2000, "shift_every": 400, "frames": 24},
    ]
    rng = random.Random(seed)
    rows = []
    for w in workloads:
        refs = generate_refs(w["pages"], w["hot"], w["length"], w["shift_every"], rng)
        feats = workload_features(w["pages"], w["hot"], w["shift_every"], w["frames"], w["length"])
        rates = {pol: simulate_paging(refs, w["frames"], pol, rng)["fault_rate"]
                 for pol in POLICIES}
        rows.append((feats, rates, refs, w["frames"]))
    return rows


def fit_learned_evictor(
    train: list[tuple[list[int], int]],
    seed: int = 0,
) -> dict[str, float]:
    """Grid-fit victim-score weights: score = w_rec*rec_rank + w_f*freq_rank,
    evict argmax. Returns best weights on training (refs, frames) pairs."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    best: dict[str, float] = {"w_rec": 0.5, "w_freq": 0.5}
    best_cost = float("inf")
    for w_rec in grid:
        for w_f in grid:
            if w_rec == 0.0 and w_f == 0.0:
                continue
            cost = sum(
                simulate_learned_eviction(refs, frames, w_rec, w_f)["faults"]
                for refs, frames in train
            )
            if cost < best_cost:
                best_cost = cost
                best = {"w_rec": w_rec, "w_freq": w_f}
    return best


def simulate_learned_eviction(
    refs: list[int], frames: int, w_rec: float, w_freq: float,
) -> dict[str, float]:
    """Learned replacement policy: score ranks KEEP-value (rank 0 =
    oldest/rarest), victim is the argmin. w_rec=1 recovers LRU, w_freq=1
    recovers LFU; the grid fit interpolates between them."""
    resident: dict[int, dict[str, float]] = {}
    faults = 0
    max_resident = 0
    time = 0
    for p in refs:
        time += 1
        if p in resident:
            resident[p]["last"] = time
            resident[p]["freq"] += 1.0
            continue
        faults += 1
        if len(resident) >= frames:
            by_rec = sorted(resident, key=lambda q: resident[q]["last"])
            by_freq = sorted(resident, key=lambda q: resident[q]["freq"])
            rank_rec = {q: i for i, q in enumerate(by_rec)}
            rank_freq = {q: i for i, q in enumerate(by_freq)}
            victim = min(resident,
                         key=lambda q: w_rec * rank_rec[q] + w_freq * rank_freq[q])
            del resident[victim]
        resident[p] = {"last": time, "freq": 1.0}
        max_resident = max(max_resident, len(resident))
    return {"faults": float(faults), "fault_rate": faults / max(len(refs), 1),
            "max_resident": float(max_resident), "refs": float(len(refs))}


@register_virtual_algorithm("paging")
class Paging(VirtualAlgorithm):
    """Demand paging virtualized: rank policies + learned evictor.

    Two fault models: one-hot ensemble MLP (primary) + per-policy
    ridge-linear (ablation baseline). virtual_step prefers the ensemble.
    """

    def __init__(
        self,
        model: PolicyFaultModel | None = None,
        fensemble: Any = None,
        evictor_weights: dict[str, float] | None = None,
        ood: FeatureDistribution | None = None,
        gate: TrustGate | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.model = model
        self.fensemble = fensemble
        self.evictor_weights = evictor_weights or {"w_rec": 0.5, "w_freq": 0.5}
        self.ood = ood or FeatureDistribution()
        self.gate = gate or TrustGate()
        self.calibrator = calibrator or Calibrator()
        self.tracker = UncertaintyTracker()

    def feature_names(self) -> list[str]:
        return list(PAGING_FEATURE_NAMES)

    def to_virtual_state(self, physical: Any) -> VirtualState:
        refs = physical.get("refs") if isinstance(physical, dict) else None
        feats = workload_features(
            int(physical.get("pages", 64)), int(physical.get("hot", 8)),
            int(physical.get("shift_every", 500)), int(physical.get("frames", 8)),
            int(physical.get("length", 1500)), refs,
        )
        return VirtualState(features=feats, feature_names=self.feature_names(),
                            metadata=dict(physical))

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        rng = rng or random.Random(0)
        refs = physical.get("refs")
        if refs is None:
            refs = generate_refs(int(physical.get("pages", 64)), int(physical.get("hot", 8)),
                                 int(physical.get("length", 1500)),
                                 int(physical.get("shift_every", 500)), rng)
        frames = int(physical.get("frames", 8))
        policy = str(intervention.parameters.get("policy", "lru"))
        if policy == "learned":
            out = simulate_learned_eviction(refs, frames, self.evictor_weights["w_rec"],
                                            self.evictor_weights["w_freq"])
        else:
            out = simulate_paging(refs, frames, policy, rng)
        out["policy"] = policy
        out["invariant_violations"] = PagingInvariants.check_outcome(
            int(out["faults"]), len(refs), int(out["max_resident"]), frames)
        return out

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        policy = str(intervention.parameters.get("policy", "lru"))
        feats = list(state.features)
        ood_score = self.ood.ood_score(feats)
        if self.fensemble is not None:
            mean, std = self.fensemble.predict(feats + policy_onehot(policy))
            rate = min(max(mean[0], 0.0), 1.0)
            uncertainty = min(max(std[0], 0.0) + 0.1 * ood_score + 0.02, 1.0)
        elif self.model is not None and self.model.weights:
            rates = self.model.predict(feats)
            rate = rates.get(policy, rates.get("lru", 0.5))
            uncertainty = min(self.model.residual_std + 0.1 * ood_score + 0.02, 1.0)
        else:
            return Prediction(outcome={}, uncertainty=0.9, ood_score=ood_score, trusted=False)
        outcome = {"fault_rate": rate}
        violations = PagingInvariants.check_prediction(rate)
        trusted, _, _ = self.gate.decide(
            ood_score, uncertainty, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome=outcome, uncertainty=uncertainty,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        return PagingInvariants.check_prediction(float(outcome.get("fault_rate", 0.0)))

    def sweep(self, physical: dict[str, Any]) -> list[dict[str, Any]]:
        # NOTE: only POLICIES have learned fault-rate models. The "learned"
        # evictor is evaluated with real runs in audit(), never hallucinated
        # through another policy's model.
        state = self.to_virtual_state(physical)
        rows = []
        for pol in POLICIES:
            pred = self.virtual_step(state, Intervention("set_policy", {"policy": pol}))
            rows.append({"policy": pol,
                         **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in pred.outcome.items()},
                         "uncertainty": round(pred.uncertainty, 4),
                         "ood": round(pred.ood_score, 4), "trusted": pred.trusted})
        return rows

    def audit(self, test_workload: dict[str, int], seed: int = 0) -> dict[str, Any]:
        """Ensemble ranking + linear ablation + per-sample uncertainty +
        learned-vs-LRU evictor comparison on the test workload."""
        from rlraft.virtual.eval import (
            bootstrap_ci, interval_coverage, ranking_consistency,
            uncertainty_validity, value_consistency,
        )

        rng = random.Random(seed)
        refs = generate_refs(test_workload["pages"], test_workload["hot"],
                             test_workload["length"], test_workload["shift_every"], rng)
        frames = test_workload["frames"]
        min_f = belady_min_faults(refs, frames)
        feats = workload_features(test_workload["pages"], test_workload["hot"],
                                  test_workload["shift_every"], frames,
                                  test_workload["length"], refs)
        predicted: dict[str, float] = {}
        linear_pred: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        abs_errs, stds = [], []
        ep_errs, ep_stds = [], []
        for pol in POLICIES:
            # hit-rate as higher-is-better score for the shared ranking harness
            samples = []
            for _ in range(10):
                r = generate_refs(test_workload["pages"], test_workload["hot"],
                                  test_workload["length"], test_workload["shift_every"], rng)
                samples.append(1.0 - simulate_paging(r, frames, pol, rng)["fault_rate"])
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[pol] = {"success_rate": sum(samples) / len(samples),
                             "success_lo": lo, "success_hi": hi}
            if self.fensemble is not None:
                mean, std = self.fensemble.predict(feats + policy_onehot(pol))
                predicted[pol] = 1.0 - min(max(mean[0], 0.0), 1.0)
                s = min(max(std[0], 0.0), 1.0)
                stds.append(s)
            elif self.model is not None and self.model.weights:
                predicted[pol] = 1.0 - self.model.predict(feats)[pol]
                s = min(self.model.residual_std + 0.02, 1.0)
                stds.append(s)
            else:
                predicted[pol] = 0.5
                s = 0.9
                stds.append(s)
            if self.model is not None and self.model.weights:
                linear_pred[pol] = 1.0 - self.model.predict(feats)[pol]
            else:
                linear_pred[pol] = 0.5
            abs_errs.append(abs(predicted[pol] - empirics[pol]["success_rate"]))
            for sample in samples:
                ep_errs.append(abs(predicted[pol] - sample))
                ep_stds.append(s)
        learned = simulate_learned_eviction(refs, frames, self.evictor_weights["w_rec"],
                                            self.evictor_weights["w_freq"])
        lru = simulate_paging(refs, frames, "lru", rng)
        return {"value": value_consistency(predicted, empirics),
                "ranking": ranking_consistency(predicted, empirics),
                "belady_min_faults": min_f,
                "learned_faults": learned["faults"], "lru_faults": lru["faults"],
                "min_gap_learned": learned["faults"] - min_f,
                "min_gap_lru": lru["faults"] - min_f,
                "test_workload": test_workload,
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

    def evaluate_evictor(
        self, workloads: list[dict[str, int]], seeds: tuple[int, ...] = (7, 11),
    ) -> dict[str, Any]:
        """Suite-level learned-vs-LRU comparison: paired fault counts per
        (workload, seed), mean difference with bootstrap CI + win rate.
        The research-grade replacement for single-workload shootouts."""
        from rlraft.virtual.eval import paired_compare

        learned_faults, lru_faults = [], []
        for w in workloads:
            for seed in seeds:
                rng = random.Random(seed)
                refs = generate_refs(w["pages"], w["hot"], w["length"],
                                     w["shift_every"], rng)
                learned_faults.append(simulate_learned_eviction(
                    refs, w["frames"], self.evictor_weights["w_rec"],
                    self.evictor_weights["w_freq"])["faults"])
                lru_faults.append(simulate_paging(refs, w["frames"], "lru", rng)["faults"])
        return {"paired_learned_minus_lru": paired_compare(learned_faults, lru_faults),
                "mean_learned": sum(learned_faults) / len(learned_faults),
                "mean_lru": sum(lru_faults) / len(lru_faults),
                "n_pairs": len(learned_faults)}
