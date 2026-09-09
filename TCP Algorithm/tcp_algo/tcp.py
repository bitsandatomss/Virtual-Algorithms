"""TCP AIMD congestion-control virtualization.

Honesty note: TCP/CUBIC/BBR/AIMD are NOT named in context.txt. This module
is a justified extension of what IS there -- "physical links / packets /
routers -> virtual network" and "an application can manipulate a socket
despite the physical network being packets moving through routers"
(context.txt ~1010-1011, ~1060). Congestion control is the canonical
algorithm living exactly at that boundary (cf. Remy/PCC/Aurora literature),
so it is the natural algo to virtualize here.

Physical (P): single-flow AIMD sender over a bottleneck link
  (base RTT, bottleneck Mbps, drop-tail buffer, random loss).
  Per-RTT: cwnd += 1 on clean RTT, cwnd = max(1, cwnd/2) on loss
  (drop-tail when queue > buf, else random loss w.p. p).
Virtual (V): ridge-linear throughput model fit from trajectories on
  [window_limited_rate, bottleneck_bw, loss, buf_norm, bias].
Intervention: do(cap=C) -- clamp cwnd <= C.
/// Conservation: cwnd in [1, cap], throughput <= bottleneck capacity.
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
from rlraft.virtual.invariants import TCPInvariants
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker

CAPS = [4, 8, 16, 32, 64]
MSS_BYTES = 1500


def simulate_aimd(
    base_rtt_ms: float,
    bw_mbps: float,
    buf_pkts: int,
    rand_loss_p: float,
    cap_pkts: int,
    rtts: int,
    rng: random.Random,
) -> dict[str, float]:
    """Explicit AIMD mechanism. Returns per-run outcome + trace stats."""
    tx_per_pkt_ms = MSS_BYTES * 8.0 / (bw_mbps * 1000.0)  # serialize one MSS
    bdp = max(base_rtt_ms / max(tx_per_pkt_ms, 1e-9), 1.0)
    cwnd = 4.0
    cwnd_trace = [cwnd]
    delivered = 0.0
    elapsed_ms = 0.0
    for _ in range(rtts):
        eff = min(cwnd, float(cap_pkts))
        queue = max(0.0, eff - bdp)
        rtt_eff = base_rtt_ms + queue * tx_per_pkt_ms
        # bottleneck serves at most bdp + buf per RTT; excess is dropped
        served = min(eff, bdp + float(buf_pkts))
        drop_tail = queue > float(buf_pkts)
        lost = drop_tail or rng.random() < rand_loss_p
        delivered += 0.0 if drop_tail else served  # overflow RTT delivers nothing
        elapsed_ms += rtt_eff
        if lost:
            cwnd = max(1.0, cwnd / 2.0)
        else:
            cwnd = min(cwnd + 1.0, float(cap_pkts))
        cwnd_trace.append(cwnd)
    thr_mbps = delivered * MSS_BYTES * 8.0 / max(elapsed_ms / 1000.0, 1e-9) / 1e6
    thr_mbps = min(thr_mbps, bw_mbps)  # service-rate conservation
    return {
        "throughput_mbps": thr_mbps,
        "loss_rate": rand_loss_p,  # configured regime loss (drop-tail folded into cwnd dynamics)
        "avg_cwnd_pkts": sum(cwnd_trace) / len(cwnd_trace),
        "elapsed_ms": elapsed_ms,
        "bdp_pkts": bdp,
    }


def bdp_of(base_rtt_ms: float, bw_mbps: float) -> float:
    """Bandwidth-delay product in packets (the dimensionless scale of AIMD)."""
    tx_per_pkt_ms = MSS_BYTES * 8.0 / max(bw_mbps * 1000.0, 1e-9)
    return max(base_rtt_ms / max(tx_per_pkt_ms, 1e-9), 1.0)


def tcp_features(
    cap_pkts: int, base_rtt_ms: float, bw_mbps: float,
    loss_p: float, buf_pkts: int,
) -> list[float]:
    """Dimensionless congestion features: AIMD dynamics scale with ratios
    to the BDP (cap/bdp, buf/bdp, loss*bdp), not raw units -- the same
    normalization that makes window-limited vs bandwidth-limited regimes
    comparable across networks. Research-grade replacement for the v1
    raw-unit features."""
    bdp = bdp_of(base_rtt_ms, bw_mbps)
    window_limited = cap_pkts * MSS_BYTES * 8.0 / max(base_rtt_ms / 1000.0, 1e-9) / 1e6
    return [
        cap_pkts / bdp,
        buf_pkts / bdp,
        loss_p * bdp,
        window_limited / max(bw_mbps, 1e-9),
        min(bdp / 64.0, 2.0),
        1.0,
    ]


TCP_FEATURE_NAMES = ["cap_bdp", "buf_bdp", "loss_bdp", "window_util", "bdp_norm", "bias"]


class AimdThroughputModel:
    """Ridge-linear surrogate fit from (features -> throughput) trajectories."""

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.weights: list[float] = []
        self.residual_std = 0.0
        self.n = 0

    def fit(self, rows: list[tuple[list[float], float]]) -> "AimdThroughputModel":
        import numpy as np

        X = np.asarray([r[0] for r in rows], dtype=float)
        y = np.asarray([r[1] for r in rows], dtype=float)
        d = X.shape[1]
        w = np.linalg.solve(
            X.T @ X + self.reg * np.eye(d), X.T @ y
        )
        self.weights = w.tolist()
        resid = y - X @ w
        self.residual_std = float(np.sqrt(max((resid @ resid) / max(len(y), 1), 0.0)))
        self.n = len(rows)
        return self

    def predict(self, feats: list[float]) -> float:
        return max(sum(w * f for w, f in zip(self.weights, feats)), 0.0)


def fit_ensemble(
    rows: list[tuple[list[float], float, dict[str, float]]],
    members: int = 5,
    epochs: int = 250,
    seed: int = 7,
) -> Any:
    """Deep-ensemble surrogate on utilization targets (thr/bw).

    Nonlinear counterpart to AimdThroughputModel (kept as the ablation
    baseline). Uncertainty = ensemble std -- measured disagreement.
    """
    from rlraft.virtual.surrogate import EnsembleRegressor

    ens = EnsembleRegressor(input_dim=len(TCP_FEATURE_NAMES), output_dim=1,
                            members=members, hidden_dim=32)
    ens.train([(f, [min(t / max(net["bw_mbps"], 1e-9), 1.0)]) for f, t, net in rows],
              epochs=epochs, seed=seed)
    return ens


def ensemble_predict(ensemble: Any, feats: list[float], bw_mbps: float) -> tuple[float, float]:
    """(throughput_mbps, uncertainty) with conservation clipping."""
    mean, std = ensemble.predict(feats)
    thr = min(max(mean[0], 0.0), 1.0) * bw_mbps
    return thr, min(max(std[0], 0.0), 1.0)


def collect_tcp_trajectories(
    grid: list[dict[str, float]] | None = None,
    episodes_per_setting: int = 8,
    rtts: int = 120,
    seed: int = 0,
) -> list[tuple[list[float], float, dict[str, float]]]:
    """(features, mean_throughput, net_params) rows over net grid x caps."""
    grid = grid or [
        {"bw_mbps": 10.0, "base_rtt_ms": 20.0, "loss_p": 0.0, "buf_pkts": 16.0},
        {"bw_mbps": 10.0, "base_rtt_ms": 100.0, "loss_p": 0.0, "buf_pkts": 64.0},
        {"bw_mbps": 50.0, "base_rtt_ms": 20.0, "loss_p": 0.01, "buf_pkts": 32.0},
        {"bw_mbps": 50.0, "base_rtt_ms": 100.0, "loss_p": 0.01, "buf_pkts": 64.0},
    ]
    rng = random.Random(seed)
    rows = []
    for net in grid:
        for cap in CAPS:
            thrs = [
                simulate_aimd(
                    net["base_rtt_ms"], net["bw_mbps"], int(net["buf_pkts"]),
                    net["loss_p"], cap, rtts, rng,
                )["throughput_mbps"]
                for _ in range(episodes_per_setting)
            ]
            feats = tcp_features(
                cap, net["base_rtt_ms"], net["bw_mbps"], net["loss_p"], int(net["buf_pkts"]),
            )
            rows.append((feats, sum(thrs) / len(thrs), dict(net)))
    return rows


@register_virtual_algorithm("tcp_congestion")
class TCPCongestion(VirtualAlgorithm):
    """AIMD virtualized: agent picks cwnd caps against a learned model.

    Two surrogates: ensemble MLP (primary) + ridge-linear (ablation
    baseline). virtual_step prefers the ensemble when fitted.
    """

    def __init__(
        self,
        model: AimdThroughputModel | None = None,
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
        return list(TCP_FEATURE_NAMES)

    def to_virtual_state(self, physical: Any) -> VirtualState:
        cap = int(physical.get("cap_pkts", 16))
        feats = tcp_features(
            cap,
            float(physical.get("base_rtt_ms", 50.0)),
            float(physical.get("bw_mbps", 25.0)),
            float(physical.get("loss_p", 0.0)),
            int(physical.get("buf_pkts", 32)),
        )
        return VirtualState(features=feats, feature_names=self.feature_names(),
                            metadata=dict(physical))

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        rng = rng or random.Random(0)
        cap = int(intervention.parameters.get("cap_pkts", 16))
        rtts = int(intervention.parameters.get("rtts", 120))
        out = simulate_aimd(
            float(physical.get("base_rtt_ms", 50.0)),
            float(physical.get("bw_mbps", 25.0)),
            int(physical.get("buf_pkts", 32)),
            float(physical.get("loss_p", 0.0)),
            cap, rtts, rng,
        )
        out["cap_pkts"] = float(cap)
        out["cwnd_pkts"] = out["avg_cwnd_pkts"]
        out["rtt_ms"] = float(physical.get("base_rtt_ms", 50.0))
        bw = float(physical.get("bw_mbps", 25.0))
        out["invariant_violations"] = TCPInvariants.check_prediction(out, bw)
        return out

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        cap = int(intervention.parameters.get("cap_pkts", 16))
        feats = tcp_features(
            cap,
            float(state.metadata.get("base_rtt_ms", 50.0)),
            float(state.metadata.get("bw_mbps", 25.0)),
            float(state.metadata.get("loss_p", 0.0)),
            int(state.metadata.get("buf_pkts", 32)),
        )
        ood_score = self.ood.ood_score(feats)
        bw = float(state.metadata.get("bw_mbps", 25.0))
        if self.ensemble is not None:
            thr, member_std = ensemble_predict(self.ensemble, feats, bw)
            uncertainty = min(member_std + 0.1 * ood_score + 0.02, 1.0)
        elif self.model is not None and self.model.weights:
            thr = min(self.model.predict(feats), bw)
            uncertainty = min(
                self.model.residual_std / max(bw, 1e-9) + 0.1 * ood_score + 0.02, 1.0)
        else:
            return Prediction(outcome={}, uncertainty=0.9, ood_score=ood_score, trusted=False)
        # Note: virtual outcome predicts throughput only; per-RTT cwnd trace
        # is audited on the physical side (check_trace), not hallucinated.
        # (check_prediction defaults an absent cwnd to the 1-MSS floor.)
        outcome = {"throughput_mbps": thr,
                   "cap_pkts": float(cap), "loss_rate": float(state.metadata.get("loss_p", 0.0)),
                   "rtt_ms": float(state.metadata.get("base_rtt_ms", 50.0))}
        violations = TCPInvariants.check_prediction(outcome, bw)
        trusted, _, _ = self.gate.decide(
            ood_score, uncertainty, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome=outcome, uncertainty=uncertainty,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        bw = float(physical.get("bw_mbps", 25.0)) if isinstance(physical, dict) else 25.0
        return TCPInvariants.check_prediction(outcome, bw)

    def sweep(self, physical: dict[str, Any]) -> list[dict[str, Any]]:
        """Counterfactual sweep over caps without running the simulator."""
        state = self.to_virtual_state(physical)
        rows = []
        for cap in CAPS:
            pred = self.virtual_step(state, Intervention("set_cwnd_cap", {"cap_pkts": cap}))
            rows.append({"cap_pkts": cap,
                         **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in pred.outcome.items()},
                         "uncertainty": round(pred.uncertainty, 4),
                         "ood": round(pred.ood_score, 4), "trusted": pred.trusted})
        return rows

    def audit(
        self, test_net: dict[str, float], episodes_per_cap: int = 12,
        rtts: int = 120, seed: int = 0,
    ) -> dict[str, Any]:
        """Ground-truth empirics per cap + ensemble value/ranking,
        linear ablation, uncertainty validity (error-correlation +
        interval coverage on per-episode samples), and cap-policy regret.
        """
        from rlraft.virtual.eval import (
            bootstrap_ci, interval_coverage, ranking_consistency,
            uncertainty_validity, value_consistency,
        )

        rng = random.Random(seed)
        predicted: dict[str, float] = {}
        linear_pred: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        state = self.to_virtual_state({**test_net, "cap_pkts": 16})
        bw = test_net["bw_mbps"]
        abs_errs, stds = [], []
        for cap in CAPS:
            key = f"cap{cap}"
            pred = self.virtual_step(state, Intervention("set_cwnd_cap", {"cap_pkts": cap}))
            predicted[key] = pred.outcome.get("throughput_mbps", 0.0) / bw
            feats = tcp_features(cap, test_net["base_rtt_ms"], bw,
                                 test_net["loss_p"], int(test_net["buf_pkts"]))
            if self.ensemble is not None:
                _, std = self.ensemble.predict(feats)
                stds.append(min(std[0], 1.0))
            else:
                stds.append(pred.uncertainty)
            if self.model is not None and self.model.weights:
                linear_pred[key] = min(self.model.predict(feats), bw) / bw
            else:
                linear_pred[key] = 0.5
            samples = [
                simulate_aimd(test_net["base_rtt_ms"], bw, int(test_net["buf_pkts"]),
                              test_net["loss_p"], cap, rtts, rng)["throughput_mbps"] / bw
                for _ in range(episodes_per_cap)
            ]
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[key] = {"success_rate": sum(samples) / len(samples),
                             "success_lo": lo, "success_hi": hi}
            mean_emp = empirics[key]["success_rate"]
            abs_errs.append(abs(predicted[key] - mean_emp))
        # per-episode uncertainty validity: 5 arms x episodes points, not 5
        ep_errs, ep_stds = [], []
        rng2 = random.Random(seed + 77)
        for cap in CAPS:
            key = f"cap{cap}"
            feats = tcp_features(cap, test_net["base_rtt_ms"], bw,
                                 test_net["loss_p"], int(test_net["buf_pkts"]))
            if self.ensemble is not None:
                _, std = self.ensemble.predict(feats)
                s = min(std[0], 1.0)
            else:
                s = 0.2
            for _ in range(episodes_per_cap):
                sample = simulate_aimd(
                    test_net["base_rtt_ms"], bw, int(test_net["buf_pkts"]),
                    test_net["loss_p"], cap, rtts, rng2)["throughput_mbps"] / bw
                ep_errs.append(abs(predicted[key] - sample))
                ep_stds.append(s)
        out: dict[str, Any] = {
            "value": value_consistency(predicted, empirics),
            "ranking": ranking_consistency(predicted, empirics),
            "test_net": test_net,
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
            },
        }
        return out

    def evaluate_policy(
        self, test_nets: list[dict[str, float]], episodes_per_cap: int = 8,
        rtts: int = 100, seed: int = 0,
    ) -> dict[str, Any]:
        """Cap-selection policy test: per net, oracle best cap (empirics)
        vs ensemble-selected vs linear-selected cap. Reports mean regret
        with bootstrap CI -- the decision-usefulness of each surrogate."""
        from rlraft.virtual.eval import bootstrap_ci

        rng = random.Random(seed)
        regrets = {"ensemble": [], "linear": []}
        for net in test_nets:
            state = self.to_virtual_state({**net, "cap_pkts": 16})
            bw = net["bw_mbps"]
            emp: dict[int, float] = {}
            for cap in CAPS:
                samples = [
                    simulate_aimd(net["base_rtt_ms"], bw, int(net["buf_pkts"]),
                                  net["loss_p"], cap, rtts, rng)["throughput_mbps"] / bw
                    for _ in range(episodes_per_cap)
                ]
                emp[cap] = sum(samples) / len(samples)
            oracle = max(emp, key=lambda c: emp[c])
            for name in ("ensemble", "linear"):
                scores = {}
                for cap in CAPS:
                    feats = tcp_features(cap, net["base_rtt_ms"], bw,
                                         net["loss_p"], int(net["buf_pkts"]))
                    if name == "ensemble" and self.ensemble is not None:
                        scores[cap], _ = ensemble_predict(self.ensemble, feats, bw)
                    elif self.model is not None and self.model.weights:
                        scores[cap] = min(self.model.predict(feats), bw)
                    else:
                        scores[cap] = 0.0
                chosen = max(scores, key=lambda c: scores[c])
                regrets[name].append(emp[oracle] - emp[chosen])
        report = {}
        for name, vals in regrets.items():
            lo, hi = bootstrap_ci(vals, resamples=1000, seed=seed)
            report[name] = {"mean_regret": sum(vals) / len(vals),
                            "ci_lo": lo, "ci_hi": hi, "n_nets": len(vals)}
        return report
