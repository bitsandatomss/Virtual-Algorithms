"""Sweep orchestrator: fit surrogate -> counterfactual sweep -> audit.

One entry point per non-Raft virtual algorithm so `virtual-sweep --algo X`
produces predictions with trust flags AND ground-truth consistency numbers
in a single report. Honest by construction: every sweep ships with its audit.
"""

from __future__ import annotations

from typing import Any


def run_sweep(algo: str, seed: int = 7) -> dict[str, Any]:
    if algo == "tcp_congestion":
        return _tcp(seed)
    if algo == "overlay_routing":
        return _routing(seed)
    if algo == "paging":
        return _paging(seed)
    if algo == "scheduling":
        return _sched(seed)
    raise ValueError(f"unknown sweep algo: {algo}")


def _tcp(seed: int) -> dict[str, Any]:
    from tcp_algo.tcp import (
        AimdThroughputModel, TCPCongestion, collect_tcp_trajectories,
        fit_ensemble,
    )

    rows = collect_tcp_trajectories(seed=seed)
    model = AimdThroughputModel().fit([(f, t) for f, t, _ in rows])
    ensemble = fit_ensemble(rows, seed=seed)
    virtual = TCPCongestion(model=model, ensemble=ensemble)
    virtual.ood.fit([f for f, _, _ in rows])
    # in-sample calibration: predicted vs empirical utilization in [0,1]
    for f, t, net in rows:
        bw = net["bw_mbps"]
        virtual.calibrator.add(min(model.predict(f) / bw, 1.0), min(t / bw, 1.0))
    test_net = {"bw_mbps": 25.0, "base_rtt_ms": 50.0, "loss_p": 0.005, "buf_pkts": 32}
    audit = virtual.audit(test_net, seed=seed)
    audit["policy"] = virtual.evaluate_policy(
        [test_net,
         {"bw_mbps": 10.0, "base_rtt_ms": 100.0, "loss_p": 0.02, "buf_pkts": 16},
         {"bw_mbps": 50.0, "base_rtt_ms": 20.0, "loss_p": 0.0, "buf_pkts": 64}],
        seed=seed)
    return {"algo": "tcp_congestion",
            "sweep": virtual.sweep({**test_net, "cap_pkts": 16}),
            "audit": audit}


def _routing(seed: int) -> dict[str, Any]:
    from routing_algo.routing import (
        RoutingSurrogate, OverlayRouting, collect_routing_trajectories,
        fit_ensemble,
    )

    paths, rows = collect_routing_trajectories(seed=seed)
    surrogate = RoutingSurrogate().fit(rows)
    ensemble = fit_ensemble(rows, seed=seed)
    virtual = OverlayRouting(surrogate=surrogate, ensemble=ensemble, paths=paths)
    virtual.ood.fit([f for f, _, _ in rows])
    for f, d, _ in rows:
        virtual.calibrator.add(surrogate.predict(f)[0], d)
    physical = {"delay_scale": 1.0, "loss_scale": 1.0}
    return {"algo": "overlay_routing",
            "sweep": virtual.sweep(physical),
            # audit on a severe loss regime: per-link ARQ(3) keeps delivery
            # saturated until links are heavily lossy, so the
            # decision-relevant test must live where paths actually differ.
            "audit": virtual.audit(delay_scale=1.0, loss_scale=10.0, seed=seed)}


def _paging(seed: int) -> dict[str, Any]:
    from paging_algo.paging import (
        PolicyFaultModel, Paging, collect_paging_trajectories,
        fit_fault_ensemble, fit_learned_evictor,
    )

    rows = collect_paging_trajectories(seed=seed)
    model = PolicyFaultModel().fit([(f, r) for f, r, _, _ in rows])
    fensemble = fit_fault_ensemble(rows, seed=seed)
    weights = fit_learned_evictor([(refs, frames) for _, _, refs, frames in rows], seed=seed)
    virtual = Paging(model=model, fensemble=fensemble, evictor_weights=weights)
    virtual.ood.fit([f for f, _, _, _ in rows])
    for f, rates, _, _ in rows:
        pred = model.predict(f)
        for pol, emp in rates.items():
            virtual.calibrator.add(1.0 - pred[pol], 1.0 - emp)
    test_workload = {"pages": 64, "hot": 12, "length": 1500, "shift_every": 400, "frames": 10}
    audit = virtual.audit(test_workload, seed=seed)
    audit["evictor_suite"] = virtual.evaluate_evictor(
        [{"pages": 64, "hot": 8, "length": 1500, "shift_every": 500, "frames": 8},
         {"pages": 64, "hot": 16, "length": 1500, "shift_every": 300, "frames": 12},
         {"pages": 128, "hot": 10, "length": 2000, "shift_every": 700, "frames": 10},
         test_workload])
    return {"algo": "paging",
            "sweep": virtual.sweep({**test_workload}),
            "audit": audit}


def _sched(seed: int) -> dict[str, Any]:
    from sched_algo.sched import (
        SchedWaitModel, Scheduling, collect_sched_trajectories, fit_ensemble,
    )

    rows = collect_sched_trajectories(seed=seed)
    model = SchedWaitModel().fit([(f, w) for f, w, _ in rows])
    ensemble = fit_ensemble(rows, seed=seed)
    virtual = Scheduling(model=model, ensemble=ensemble)
    virtual.ood.fit([f for f, _, _ in rows])
    scale = max(w for _, w, _ in rows)
    for f, w, _ in rows:
        virtual.calibrator.add(min(model.predict(f) / scale, 1.0), min(w / scale, 1.0))
    load = {"arrival_rate": 0.05, "mean_burst_ms": 12.0, "burst_cv": 0.5}  # rho = 0.6
    return {"algo": "scheduling",
            "sweep": virtual.sweep({**load, "discipline": "fcfs"}),
            "audit": virtual.audit(load, seed=seed)}
