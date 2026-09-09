"""VirtualLab: the closed scientific loop over a virtual algorithm.

context.txt loop (lines ~442-479, 615):

    REAL WORLD -> experiments/sensors -> SURROGATE (learned dynamics)
      -> virtual experiments -> hypothesis/design/prediction
      -> REAL EXPERIMENT -> new data -> surrogate ...

  observe -> learn -> simulate -> identify uncertainty
    -> choose experiment -> observe

The simulator plays the role of Reality (ground truth); the surrogate is
the learned world dynamics; TrustGate decides when virtual results may
guide the next real experiment; active selection queries the most
uncertain intervention first.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from rlraft.virtual.algorithms import RaftElection
from rlraft.virtual.base import Intervention, MaturityLevel
from rlraft.virtual.surrogate import DynamicsSurrogate, TrajectoryDataset


@dataclass(slots=True)
class LabRoundResult:
    round_index: int
    best_intervention: Intervention
    predicted: dict[str, float]
    observed: dict[str, Any]
    trusted: bool
    mean_ood: float
    surrogate_mae: float
    maturity: MaturityLevel


@dataclass
class VirtualLab:
    nodes: int = 20
    seed: int = 7
    virtual: RaftElection = field(default_factory=RaftElection)
    surrogate_path: str = "runs/virtual/surrogate"

    def _candidate_arms(self) -> list[str]:
        return list(RaftElection.ARMS)

    def bootstrap(
        self,
        nodes_list: list[int] | None = None,
        episodes_per_setting: int = 40,
        epochs: int = 250,
    ) -> dict[str, float]:
        dataset = TrajectoryDataset().collect(
            nodes_list=nodes_list or [5, self.nodes],
            episodes_per_setting=episodes_per_setting,
            seed=self.seed,
        )
        surrogate = DynamicsSurrogate()
        metrics = surrogate.train(dataset, epochs=epochs, seed=self.seed)
        self.virtual.surrogate = surrogate
        # fit OOD distribution on canonical in-distribution states
        from rlraft.virtual.surrogate import cluster_features

        rows = []
        for arm_index in range(5):
            rows.append(cluster_features(self.nodes, 120.0, 0.02, 0.2, False, arm_index))
        self.virtual.ood.fit(rows)
        # calibrate on the training rows (predicted vs empirical success)
        for feats, outcome in dataset.rows:
            pred = surrogate.predict(feats)
            self.virtual.calibrator.add(pred["success_prob"], outcome["success_prob"])
        return metrics

    def run_round(
        self,
        round_index: int,
        rng: random.Random,
        mean_rtt_ms: float = 120.0,
        mean_loss: float = 0.02,
        mean_log_gap: float = 0.2,
    ) -> LabRoundResult:
        """One active loop: virtual sweep -> pick most uncertain trusted-or-
        informative arm -> real experiment -> feed calibration."""
        physical = {
            "nodes": self.nodes, "mean_rtt_ms": mean_rtt_ms,
            "mean_loss": mean_loss, "mean_log_gap": mean_log_gap,
            "recent_split": False,
        }
        state = self.virtual.to_virtual_state(physical)
        # virtual experiments over all arms
        scored: list[tuple[str, Any]] = []
        for arm in self._candidate_arms():
            pred = self.virtual.virtual_step(state, Intervention("set_timeout_arm", {"arm": arm}))
            scored.append((arm, pred))
        # active selection: highest uncertainty first (info value),
        # preferring trusted predictions on ties
        scored.sort(key=lambda kv: (kv[1].trusted, -kv[1].uncertainty), reverse=False)
        # pick max uncertainty among trusted; else max uncertainty overall
        trusted = [s for s in scored if s[1].trusted]
        pool = trusted or scored
        best_arm, best_pred = max(pool, key=lambda kv: kv[1].uncertainty)
        intervention = Intervention("set_timeout_arm", {"arm": best_arm})
        observed = self.virtual.physical_step(physical, intervention, rng)
        # feed back: calibration + OOD still in-dist
        self.virtual.calibrator.add(
            best_pred.outcome.get("success_prob", 0.5), 1.0 if observed["success"] else 0.0,
        )
        mean_ood = sum(p.ood_score for _, p in scored) / len(scored)
        maturity = (
            MaturityLevel.SCIENTIFIC
            if best_pred.trusted and len(self.virtual.calibrator.pairs) >= 12 and round_index >= 1
            else MaturityLevel.COUNTERFACTUAL if best_pred.trusted
            else MaturityLevel.EXEMPLAR
        )
        return LabRoundResult(
            round_index=round_index,
            best_intervention=intervention,
            predicted=dict(best_pred.outcome),
            observed=observed,
            trusted=best_pred.trusted,
            mean_ood=mean_ood,
            surrogate_mae=self.virtual.calibrator.ece(),
            maturity=maturity,
        )

    def run(
        self,
        rounds: int = 3,
        episodes_per_setting: int = 40,
        epochs: int = 250,
    ) -> dict[str, Any]:
        boot = self.bootstrap(episodes_per_setting=episodes_per_setting, epochs=epochs)
        rng = random.Random(self.seed + 999)
        results: list[LabRoundResult] = []
        for i in range(rounds):
            results.append(self.run_round(i, rng))
        return {
            "bootstrap": boot,
            "rounds": [
                {
                    "round": r.round_index,
                    "arm": r.best_intervention.parameters.get("arm"),
                    "predicted_success": r.predicted.get("success_prob"),
                    "observed_success": r.observed.get("success"),
                    "observed_time_ms": r.observed.get("total_time_ms"),
                    "trusted": r.trusted,
                    "maturity": r.maturity.name,
                }
                for r in results
            ],
            "final_ece": self.virtual.calibrator.ece(),
            "final_maturity": results[-1].maturity.name if results else MaturityLevel.EXEMPLAR.name,
        }
