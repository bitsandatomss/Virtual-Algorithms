"""Unified cross-algorithm matrix: one bar for every virtual algorithm.

Each entry is fit + audited + OOD-benchmarked at multiple seeds. Metrics
are the shared harness ones (value MAE/CI-coverage, ranking rho/regret)
plus a Mahalanobis AUROC separating train-regime from stress-regime
features, and a maturity assignment per MATURITY_GATES. Backoff is capped
at INTERPOLATIVE with an explicit reason (lookup table, no decision
semantics). Raft is excluded by default (torch ensemble cost) with its
reference numbers cited in docs; pass include_raft=True to run it light.
"""

from __future__ import annotations

import random
from typing import Any

from rlraft.virtual.eval import MATURITY_GATES, ood_benchmark
from rlraft.virtual.trust import MahalanobisOOD


def _summarize(values: list[float]) -> dict[str, float]:
    return {"mean": sum(values) / max(len(values), 1),
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0}


def _maturity(mean_rho: float, mean_regret_frac: float, auroc: float,
              cap: str | None = None) -> tuple[str, str]:
    l3 = (mean_rho >= MATURITY_GATES["rho_counterfactual"]
          and mean_regret_frac <= MATURITY_GATES["regret_fraction"])
    l4 = l3 and auroc >= MATURITY_GATES["auroc_mechanistic"]
    level = "MECHANISTIC" if l4 else "COUNTERFACTUAL" if l3 else "INTERPOLATIVE"
    reason = "gates pass" if l4 else (
        "ranking gates pass; OOD separation below L4 bar" if l3
        else "ranking or regret gate fails")
    if cap == "INTERPOLATIVE" and level in ("COUNTERFACTUAL", "MECHANISTIC"):
        return "INTERPOLATIVE", "capped: no decision semantics"
    if cap == "INTERPOLATIVE" and level == "INTERPOLATIVE":
        return level, "capped: no decision semantics; " + reason
    return level, reason


def _auroc(train_feats: list[list[float]], ood_feats: list[list[float]]) -> dict[str, float]:
    det = MahalanobisOOD()
    det.fit(train_feats)
    return ood_benchmark(det, train_feats, ood_feats)


def _backoff(seed: int, light: bool):
    import random as _r

    from rlraft.virtual.algorithms import Backoff

    n = 10 if light else 30
    vb = Backoff()
    rng = _r.Random(seed)
    vb.fit([(f, vb.physical_delay(f, rng)) for f in range(9) for _ in range(n)])
    audit = vb.audit(seed=seed, samples_per_bucket=10 if light else 20)
    train_feats = [vb.to_virtual_state({"failures": f}).features for f in range(9)]
    ood_feats = [vb.to_virtual_state({"failures": f}).features for f in range(10, 15)]
    return audit, train_feats, ood_feats, "INTERPOLATIVE"


def _gossip(seed: int, light: bool):
    import random as _r

    from rlraft.virtual.algorithms import Gossip
    from rlraft.virtual.base import Intervention

    eps = 6 if light else 12
    g = Gossip(nodes=20)
    rng = _r.Random(seed)
    # Fit in the interior regime (fanout 2, 3 rounds): at fanout 3 / 5
    # rounds coverage saturates at 1.0 and p is unidentifiable from below
    # (any p above threshold fits perfectly). The audit then tests whether
    # the fitted p generalizes to the saturating regime across fanouts.
    obs = [g.physical_step({}, Intervention(
        "spread", {"fanout": 2, "rounds": 3, "p": 0.7}), rng)["coverage"]
        for _ in range(10)]
    g.fit_p(obs, fanout=2, rounds=3)
    audit = g.audit(rounds=5, p_true=0.7, episodes_per_fanout=eps, seed=seed)
    train_feats = [g.to_virtual_state({"fanout": f}).features for f in (1, 2, 3, 4, 6)]
    g_ood = Gossip(nodes=20)
    ood_feats = [g_ood.to_virtual_state({"fanout": f}).features for f in (10, 12, 16)]
    return audit, train_feats, ood_feats, None


def _tcp(seed: int, light: bool):
    from tcp_algo.tcp import AimdThroughputModel, TCPCongestion, collect_tcp_trajectories, fit_ensemble, tcp_features

    eps, rtts = (4, 40) if light else (8, 120)
    rows = collect_tcp_trajectories(episodes_per_setting=eps, rtts=rtts, seed=seed)
    model = AimdThroughputModel().fit([(f, t) for f, t, _ in rows])
    members, epochs = (3, 60) if light else (5, 250)
    ensemble = fit_ensemble(rows, members=members, epochs=epochs, seed=seed)
    virtual = TCPCongestion(model=model, ensemble=ensemble)
    virtual.ood.fit([f for f, _, _ in rows])
    for f, t, net in rows:
        virtual.calibrator.add(min(model.predict(f) / net["bw_mbps"], 1.0), min(t / net["bw_mbps"], 1.0))
    test_net = {"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32}
    audit = virtual.audit(test_net, episodes_per_cap=6 if light else 12,
                          rtts=rtts, seed=seed)
    ood_net = {"bw_mbps": 100.0, "base_rtt_ms": 200.0, "loss_p": 0.05, "buf_pkts": 128}
    ood_feats = [tcp_features(c, ood_net["base_rtt_ms"], ood_net["bw_mbps"],
                              ood_net["loss_p"], ood_net["buf_pkts"]) for c in (4, 8, 16, 32, 64)]
    return audit, [f for f, _, _ in rows], ood_feats, None


def _routing(seed: int, light: bool):
    from routing_algo.routing import (
        RoutingSurrogate, OverlayRouting, build_topology,
        collect_routing_trajectories, fit_ensemble,
    )

    packets = 60 if light else 200
    paths, rows = collect_routing_trajectories(packets=packets, seed=seed)
    surrogate = RoutingSurrogate().fit(rows)
    members, epochs = (3, 60) if light else (5, 250)
    ensemble = fit_ensemble(rows, members=members, epochs=epochs, seed=seed)
    virtual = OverlayRouting(surrogate=surrogate, ensemble=ensemble, paths=paths)
    virtual.ood.fit([f for f, _, _ in rows])
    for f, d, _ in rows:
        virtual.calibrator.add(surrogate.predict(f)[0], d)
    audit = virtual.audit(delay_scale=1.0, loss_scale=10.0, packets=packets, seed=seed)
    ood_topo = build_topology(0.3, 14.0)
    ood_feats = [virtual._feats_for(ood_topo, p) for p in paths]
    return audit, [f for f, _, _ in rows], ood_feats, None


def _paging(seed: int, light: bool):
    from paging_algo.paging import (
        PolicyFaultModel, Paging, collect_paging_trajectories,
        fit_fault_ensemble, fit_learned_evictor, workload_features,
    )

    if light:
        # two workloads minimum: Mahalanobis needs >=2 rows for a
        # non-degenerate covariance (a single-row fit inverts the detector)
        workloads = [{"pages": 32, "hot": 6, "length": 400, "shift_every": 200, "frames": 6},
                     {"pages": 32, "hot": 10, "length": 400, "shift_every": 100, "frames": 8}]
        test_w = {"pages": 32, "hot": 8, "length": 400, "shift_every": 200, "frames": 6}
    else:
        workloads = None
        test_w = {"pages": 64, "hot": 12, "length": 1500, "shift_every": 400, "frames": 10}
    rows = collect_paging_trajectories(workloads=workloads, seed=seed)
    model = PolicyFaultModel().fit([(f, r) for f, r, _, _ in rows])
    members, epochs = (3, 60) if light else (5, 250)
    fensemble = fit_fault_ensemble(rows, members=members, epochs=epochs, seed=seed)
    weights = fit_learned_evictor([(refs, fr) for _, _, refs, fr in rows], seed=seed)
    virtual = Paging(model=model, fensemble=fensemble, evictor_weights=weights)
    virtual.ood.fit([f for f, _, _, _ in rows])
    for f, rates, _, _ in rows:
        pred = model.predict(f)
        for pol, emp in rates.items():
            virtual.calibrator.add(1.0 - pred[pol], 1.0 - emp)
    audit = virtual.audit(test_w, seed=seed)
    ood_feats = [workload_features(256, 96, 100, 16, 1500)]
    return audit, [f for f, _, _, _ in rows], ood_feats, None


def _sched(seed: int, light: bool):
    from sched_algo.sched import (
        SchedWaitModel, Scheduling, collect_sched_trajectories, fit_ensemble, sched_features,
    )

    jobs, reps = (40, 3) if light else (120, 8)
    rows = collect_sched_trajectories(seed=seed)
    model = SchedWaitModel().fit([(f, w) for f, w, _ in rows])
    members, epochs = (3, 60) if light else (5, 250)
    ensemble = fit_ensemble(rows, members=members, epochs=epochs, seed=seed)
    virtual = Scheduling(model=model, ensemble=ensemble)
    virtual.ood.fit([f for f, _, _ in rows])
    scale = max(w for _, w, _ in rows)
    for f, w, _ in rows:
        virtual.calibrator.add(min(model.predict(f) / scale, 1.0), min(w / scale, 1.0))
    load = {"arrival_rate": 0.05, "mean_burst_ms": 12.0}
    audit = virtual.audit(load, jobs=jobs, repeats=reps, seed=seed)
    ood_feats = [sched_features(0.05 * 12.0 * 2.4, 12.0, q, b, 0.5)
                 for q, b in ((float("inf"), 0.0), (float("inf"), 1.0), (1.0, 0.0), (4.0, 0.0))]
    return audit, [f for f, _, _ in rows], ood_feats, None


_BUILDERS = {
    "backoff": _backoff,
    "gossip": _gossip,
    "tcp_congestion": _tcp,
    "overlay_routing": _routing,
    "paging": _paging,
    "scheduling": _sched,
}


def run_matrix(
    algos: list[str] | None = None,
    seeds: tuple[int, ...] = (7, 11, 13),
    light: bool = False,
    include_raft: bool = False,
) -> dict[str, Any]:
    """Fit + audit + OOD-benchmark every listed algo at every seed."""
    algos = algos or sorted(_BUILDERS)
    matrix: dict[str, Any] = {}
    for name in algos:
        builder = _BUILDERS[name]
        rhos, regrets, maes, covs, aurocs = [], [], [], [], []
        per_seed = []
        cap = None
        for seed in seeds:
            audit, train_feats, ood_feats, cap = builder(seed, light)
            rhos.append(audit["ranking"]["spearman_rho"])
            regrets.append(audit["ranking"]["regret_fraction"])
            maes.append(audit["value"]["mae"])
            covs.append(audit["value"]["ci_coverage"])
            auroc = _auroc(train_feats, ood_feats)["auroc"]
            aurocs.append(auroc)
            per_seed.append({"seed": seed,
                             "rho": audit["ranking"]["spearman_rho"],
                             "regret_fraction": audit["ranking"]["regret_fraction"],
                             "mae": audit["value"]["mae"],
                             "ci_coverage": audit["value"]["ci_coverage"],
                             "auroc": auroc})
        mean_rho = sum(rhos) / len(rhos)
        mean_reg = sum(regrets) / len(regrets)
        mean_auroc = sum(aurocs) / len(aurocs)
        level, reason = _maturity(mean_rho, mean_reg, mean_auroc, cap)
        matrix[name] = {"rho": _summarize(rhos), "regret_fraction": _summarize(regrets),
                        "mae": _summarize(maes), "ci_coverage": _summarize(covs),
                        "auroc": _summarize(aurocs), "maturity": level,
                        "maturity_reason": reason, "per_seed": per_seed}
    if include_raft:
        from rlraft.virtual.eval import run_audit

        for seed in seeds:
            rep = run_audit(episodes_per_arm=10, ensemble_members=2, epochs=20,
                            seed=seed, stress_loss=False)
            matrix.setdefault("raft_election", {"per_seed": []})["per_seed"].append(
                {"seed": seed, "rho": rep["ranking_consistency"]["spearman_rho"],
                 "regret_fraction": rep["ranking_consistency"]["regret_fraction"],
                 "mae": rep["heldout_mae"],
                 "ci_coverage": rep["value_consistency"]["ci_coverage"],
                 "auroc": rep["ood_benchmark"]["auroc"]})
        r = matrix["raft_election"]["per_seed"]
        matrix["raft_election"].update(
            {"rho": _summarize([p["rho"] for p in r]),
             "regret_fraction": _summarize([p["regret_fraction"] for p in r]),
             "mae": _summarize([p["mae"] for p in r]),
             "ci_coverage": _summarize([p["ci_coverage"] for p in r]),
             "auroc": _summarize([p["auroc"] for p in r])})
        m_rho = matrix["raft_election"]["rho"]["mean"]
        m_reg = matrix["raft_election"]["regret_fraction"]["mean"]
        m_au = matrix["raft_election"]["auroc"]["mean"]
        level, reason = _maturity(m_rho, m_reg, m_au)
        matrix["raft_election"].update({"maturity": level, "maturity_reason": reason})
    return {"seeds": list(seeds), "gates": MATURITY_GATES, "matrix": matrix,
            "note": "raft excluded by default (torch cost); reference seed-7 full audit in docs/CRITICAL_ANALYSIS.md"}
