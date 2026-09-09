"""Overlay routing: packets -> socket/overlay virtualization.

Directly grounded in context.txt: "physical links / packets / routers ->
virtual network / logical circuit / overlay" (~1010-1011), "an application
can manipulate a socket despite the physical network being packets moving
through routers" (~1060), "the application doesn't operate at
packet-routing level" (~1500).

Physical (P): fixed 6-node topology with per-link (delay, loss); per-link
ARQ (up to 3 attempts); candidate loop-free overlay paths src -> dst.
Virtual (V): ridge-linear models of delivery_rate and latency fit from
trajectories over randomized link conditions.
/// Conservation: latency >= propagation lower bound, delivery in [0,1],
paths loop-free and connected.
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
from rlraft.virtual.invariants import RoutingInvariants
from rlraft.virtual.trust import Calibrator, FeatureDistribution, TrustGate, UncertaintyTracker

Link = tuple[int, float, float]  # (neighbor, delay_ms, loss_p)


def build_topology(delay_scale: float = 1.0, loss_scale: float = 1.0) -> dict[int, list[Link]]:
    base: dict[int, list[tuple[int, float, float]]] = {
        0: [(1, 10.0, 0.01), (2, 25.0, 0.005)],
        1: [(0, 10.0, 0.01), (3, 15.0, 0.02), (4, 40.0, 0.005)],
        2: [(0, 25.0, 0.005), (3, 10.0, 0.01), (5, 30.0, 0.02)],
        3: [(1, 15.0, 0.02), (2, 10.0, 0.01), (4, 12.0, 0.01), (5, 18.0, 0.03)],
        4: [(1, 40.0, 0.005), (3, 12.0, 0.01), (5, 8.0, 0.005)],
        5: [(2, 30.0, 0.02), (3, 18.0, 0.03), (4, 8.0, 0.005)],
    }
    topo: dict[int, list[Link]] = {}
    for node, links in base.items():
        topo[node] = [(nb, d * delay_scale, min(l * loss_scale, 0.4)) for nb, d, l in links]
    return topo


def enumerate_paths(
    topo: dict[int, list[Link]], src: int, dst: int, k: int = 4, max_depth: int = 5,
) -> list[list[int]]:
    found: list[tuple[float, list[int]]] = []

    def dfs(node: int, path: list[int], delay: float) -> None:
        if len(path) > max_depth:
            return
        if node == dst:
            found.append((delay, list(path)))
            return
        for nb, d, _ in topo.get(node, []):
            if nb in path:
                continue
            path.append(nb)
            dfs(nb, path, delay + d)
            path.pop()

    dfs(src, [src], 0.0)
    found.sort(key=lambda t: t[0])
    return [p for _, p in found[:k]]


def path_link_params(topo: dict[int, list[Link]], path: list[int]) -> list[tuple[float, float]]:
    params = []
    for a, b in zip(path, path[1:]):
        for nb, d, l in topo[a]:
            if nb == b:
                params.append((d, l))
                break
    return params


def simulate_path(
    topo: dict[int, list[Link]], path: list[int], packets: int, rng: random.Random,
) -> dict[str, float]:
    """Explicit per-packet mechanism with per-link ARQ (3 attempts)."""
    params = path_link_params(topo, path)
    prop_bound = sum(d for d, _ in params)
    delivered_lat: list[float] = []
    delivered = 0
    for _ in range(packets):
        lat = 0.0
        ok = True
        for d, l in params:
            attempt = 0
            while True:
                lat += d * rng.uniform(0.8, 1.2)
                attempt += 1
                if rng.random() >= l:
                    break
                if attempt >= 3:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            delivered += 1
            delivered_lat.append(lat)
    return {
        "delivery_rate": delivered / max(packets, 1),
        "mean_latency_ms": sum(delivered_lat) / max(len(delivered_lat), 1),
        "prop_bound_ms": prop_bound,
    }


def routing_features(path_len: int, delay_sum: float, bottleneck_loss: float,
                     loss_mass: float, delay_per_hop: float = 0.0,
                     bottleneck_share: float = 0.0) -> list[float]:
    """Path features with loss-concentration structure: where loss sits
    matters as much as how much (a single bad link vs diffuse loss behave
    differently under per-link ARQ). Research-grade extension of the v1
    four-feature set."""
    return [float(path_len), delay_sum, bottleneck_loss, loss_mass,
            delay_per_hop, bottleneck_share, 1.0]


ROUTING_FEATURE_NAMES = ["hops", "delay_sum_ms", "bottleneck_loss", "loss_mass",
                         "delay_per_hop", "bottleneck_share", "bias"]


def _path_stats(topo: dict[int, list[Link]], path: list[int]) -> tuple[float, float, float, float, float]:
    params = path_link_params(topo, path)
    delay_sum = sum(d for d, _ in params)
    bottleneck = max([l for _, l in params] or [0.0])
    prod = 1.0
    for _, l in params:
        prod *= 1.0 - l
    mass = 1.0 - prod
    hops = max(len(params), 1)
    share = bottleneck / max(mass, 1e-9)  # 1.0 = all loss on one link
    return delay_sum, bottleneck, mass, delay_sum / hops, min(share, 2.0)


class RoutingSurrogate:
    """Ridge-linear delivery + latency models fit from trajectories."""

    def __init__(self, reg: float = 1e-3) -> None:
        self.reg = reg
        self.w_delivery: list[float] = []
        self.w_latency: list[float] = []
        self.resid_delivery = 0.0
        self.resid_latency = 0.0

    def fit(self, rows: list[tuple[list[float], float, float]]) -> "RoutingSurrogate":
        import numpy as np

        X = np.asarray([r[0] for r in rows], dtype=float)
        yd = np.asarray([r[1] for r in rows], dtype=float)
        yl = np.asarray([r[2] for r in rows], dtype=float)
        d = X.shape[1]
        A = X.T @ X + self.reg * np.eye(d)
        self.w_delivery = np.linalg.solve(A, X.T @ yd).tolist()
        self.w_latency = np.linalg.solve(A, X.T @ yl).tolist()
        rd = yd - X @ np.asarray(self.w_delivery)
        rl = yl - X @ np.asarray(self.w_latency)
        self.resid_delivery = float(np.sqrt(max(rd @ rd / max(len(yd), 1), 0.0)))
        self.resid_latency = float(np.sqrt(max(rl @ rl / max(len(yl), 1), 0.0)))
        return self

    def predict(self, feats: list[float]) -> tuple[float, float]:
        import numpy as np

        x = np.asarray(feats, dtype=float)
        delivery = float(np.clip(x @ np.asarray(self.w_delivery), 0.0, 1.0))
        latency = float(max(x @ np.asarray(self.w_latency), 0.0))
        return delivery, latency


def fit_ensemble(
    rows: list[tuple[list[float], float, float]],
    members: int = 5,
    epochs: int = 250,
    seed: int = 7,
    latency_scale: float = 100.0,
) -> Any:
    """Deep-ensemble surrogate on (delivery, latency/scale) vector targets.

    Nonlinear counterpart to RoutingSurrogate (kept as ablation baseline).
    """
    from rlraft.virtual.surrogate import EnsembleRegressor

    ens = EnsembleRegressor(input_dim=len(ROUTING_FEATURE_NAMES), output_dim=2,
                            members=members, hidden_dim=32)
    ens.train([(f, [d, t / latency_scale]) for f, d, t in rows],
              epochs=epochs, seed=seed)
    return ens


def ensemble_predict(ensemble: Any, feats: list[float],
                     latency_scale: float = 100.0) -> tuple[float, float, float, float]:
    """(delivery, latency_ms, delivery_std, latency_std)."""
    mean, std = ensemble.predict(feats)
    return (min(max(mean[0], 0.0), 1.0), max(mean[1], 0.0) * latency_scale,
            min(max(std[0], 0.0), 1.0), max(std[1], 0.0) * latency_scale)


def collect_routing_trajectories(
    scales: list[tuple[float, float]] | None = None,
    packets: int = 200, seed: int = 0,
) -> tuple[list[list[int]], list[tuple[list[float], float, float]]]:
    """Trajectories over a loss gradient (benign -> harsh) x paths.

    Training on the gradient -- not a single benign point -- is what makes
    the harsh-regime audit a generalization test with teeth rather than a
    foregone extrapolation failure. Note ARQ(3) keeps delivery saturated
    until severe loss, so the gradient spans ~1.0 down to ~0.97; latency
    carries most of the path signal (see audit + CROSS_ALGO_ANALYSIS).
    """
    scales = scales or [(1.0, 0.5), (1.0, 1.0), (1.5, 2.0), (1.0, 4.0), (1.0, 6.0)]
    rng = random.Random(seed)
    paths = enumerate_paths(build_topology(), 0, 5)
    rows = []
    for ds, ls in scales:
        topo = build_topology(ds, ls)
        for path in paths:
            delay_sum, bottleneck, mass, per_hop, share = _path_stats(topo, path)
            feats = routing_features(len(path) - 1, delay_sum, bottleneck, mass,
                                     per_hop, share)
            out = simulate_path(topo, path, packets, rng)
            rows.append((feats, out["delivery_rate"], out["mean_latency_ms"]))
    return paths, rows


@register_virtual_algorithm("overlay_routing")
class OverlayRouting(VirtualAlgorithm):
    """Packets virtualized: agent picks overlay paths via learned model.

    Two surrogates: vector ensemble MLP (primary) + ridge-linear
    (ablation baseline). virtual_step prefers the ensemble when fitted.
    """

    def __init__(
        self,
        surrogate: RoutingSurrogate | None = None,
        ensemble: Any = None,
        paths: list[list[int]] | None = None,
        ood: FeatureDistribution | None = None,
        gate: TrustGate | None = None,
        calibrator: Calibrator | None = None,
    ) -> None:
        self.surrogate = surrogate
        self.ensemble = ensemble
        self.paths = paths if paths is not None else enumerate_paths(build_topology(), 0, 5)
        self.ood = ood or FeatureDistribution()
        self.gate = gate or TrustGate()
        self.calibrator = calibrator or Calibrator()
        self.tracker = UncertaintyTracker()

    def feature_names(self) -> list[str]:
        return list(ROUTING_FEATURE_NAMES)

    def _feats_for(self, topo: dict[int, list[Link]], path: list[int]) -> list[float]:
        delay_sum, bottleneck, mass, per_hop, share = _path_stats(topo, path)
        return routing_features(len(path) - 1, delay_sum, bottleneck, mass,
                                per_hop, share)

    def to_virtual_state(self, physical: Any) -> VirtualState:
        topo = build_topology(float(physical.get("delay_scale", 1.0)),
                              float(physical.get("loss_scale", 1.0)))
        idx = int(physical.get("path_index", 0)) % len(self.paths)
        feats = self._feats_for(topo, self.paths[idx])
        meta = dict(physical)
        meta["topo"] = topo
        return VirtualState(features=feats, feature_names=self.feature_names(), metadata=meta)

    def physical_step(self, physical: Any, intervention: Intervention, rng: Any) -> dict[str, Any]:
        rng = rng or random.Random(0)
        topo = build_topology(float(physical.get("delay_scale", 1.0)),
                              float(physical.get("loss_scale", 1.0)))
        idx = int(intervention.parameters.get("path_index", 0)) % len(self.paths)
        path = self.paths[idx]
        packets = int(intervention.parameters.get("packets", 200))
        out = simulate_path(topo, path, packets, rng)
        edges = {(a, nb) for a, links in topo.items() for nb, _, _ in links}
        violations = RoutingInvariants.check_prediction(out, out["prop_bound_ms"], path)
        violations += RoutingInvariants.check_path_connected(path, edges)
        return {"path_index": idx, "path": path, **out, "invariant_violations": violations}

    def virtual_step(self, state: VirtualState, intervention: Intervention) -> Prediction:
        idx = int(intervention.parameters.get("path_index", 0)) % len(self.paths)
        path = self.paths[idx]
        topo = state.metadata.get("topo") or build_topology()
        feats = self._feats_for(topo, path)
        ood_score = self.ood.ood_score(feats)
        prop = sum(d for d, _ in path_link_params(topo, path))
        if self.ensemble is not None:
            delivery, latency, d_std, _l_std = ensemble_predict(self.ensemble, feats)
            latency = max(latency, prop)  # conservation at prediction time
            uncertainty = min(d_std + 0.1 * ood_score + 0.02, 1.0)
        elif self.surrogate is not None and self.surrogate.w_delivery:
            delivery, latency = self.surrogate.predict(feats)
            latency = max(latency, prop)
            uncertainty = min(self.surrogate.resid_delivery + 0.1 * ood_score + 0.02, 1.0)
        else:
            return Prediction(outcome={}, uncertainty=0.9, ood_score=ood_score, trusted=False)
        outcome = {"delivery_rate": delivery, "mean_latency_ms": latency}
        violations = RoutingInvariants.check_prediction(outcome, prop, path)
        trusted, _, _ = self.gate.decide(
            ood_score, uncertainty, self.calibrator.ece(), violations, is_intervention=True)
        return Prediction(outcome=outcome, uncertainty=uncertainty,
                          ood_score=ood_score, trusted=trusted)

    def check_invariants(self, physical: Any, outcome: dict[str, Any]) -> list[str]:
        path = list(physical.get("path", self.paths[0])) if isinstance(physical, dict) else self.paths[0]
        return RoutingInvariants.check_prediction(outcome, 0.0, path)

    def sweep(self, physical: dict[str, Any]) -> list[dict[str, Any]]:
        state = self.to_virtual_state(physical)
        rows = []
        for i, path in enumerate(self.paths):
            pred = self.virtual_step(state, Intervention("set_path", {"path_index": i}))
            rows.append({"path_index": i, "path": path,
                         **{k: round(v, 4) if isinstance(v, float) else v
                            for k, v in pred.outcome.items()},
                         "uncertainty": round(pred.uncertainty, 4),
                         "ood": round(pred.ood_score, 4), "trusted": pred.trusted})
        return rows

    def audit(self, delay_scale: float = 1.0, loss_scale: float = 1.0,
              packets: int = 200, seed: int = 0) -> dict[str, Any]:
        """Ensemble value/ranking + linear ablation + per-episode
        uncertainty validity (delivery only)."""
        from rlraft.virtual.eval import (
            bootstrap_ci, interval_coverage, ranking_consistency,
            uncertainty_validity, value_consistency,
        )

        rng = random.Random(seed)
        topo = build_topology(delay_scale, loss_scale)
        predicted: dict[str, float] = {}
        linear_pred: dict[str, float] = {}
        empirics: dict[str, dict[str, float]] = {}
        abs_errs, stds = [], []
        for i, path in enumerate(self.paths):
            key = f"path{i}"
            feats = self._feats_for(topo, path)
            if self.ensemble is not None:
                delivery, _, d_std, _ = ensemble_predict(self.ensemble, feats)
                stds.append(min(d_std, 1.0))
            elif self.surrogate is not None and self.surrogate.w_delivery:
                delivery, _ = self.surrogate.predict(feats)
                stds.append(min(self.surrogate.resid_delivery + 0.02, 1.0))
            else:
                delivery = 0.5
                stds.append(0.9)
            predicted[key] = delivery
            if self.surrogate is not None and self.surrogate.w_delivery:
                linear_pred[key], _ = self.surrogate.predict(feats)
            else:
                linear_pred[key] = 0.5
            samples = [simulate_path(topo, path, packets, rng)["delivery_rate"]
                       for _ in range(12)]
            lo, hi = bootstrap_ci(samples, resamples=500, seed=seed)
            empirics[key] = {"success_rate": sum(samples) / len(samples),
                             "success_lo": lo, "success_hi": hi}
            abs_errs.append(abs(predicted[key] - empirics[key]["success_rate"]))
        # per-packet-batch uncertainty validity: 4 paths x 12 samples
        ep_errs, ep_stds = [], []
        rng2 = random.Random(seed + 77)
        for i, path in enumerate(self.paths):
            key = f"path{i}"
            s = stds[i]
            for _ in range(12):
                sample = simulate_path(topo, path, packets, rng2)["delivery_rate"]
                ep_errs.append(abs(predicted[key] - sample))
                ep_stds.append(s)
        return {"value": value_consistency(predicted, empirics),
                "ranking": ranking_consistency(predicted, empirics),
                "regime": {"delay_scale": delay_scale, "loss_scale": loss_scale},
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
