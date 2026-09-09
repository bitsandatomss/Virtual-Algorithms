"""Rigorous evaluation harness: every theoretical claim gets a number.

Implements docs/THEORY.md measurements:
  - value_consistency  (§3a): per-arm MAE + Wilson CIs
  - ranking_consistency (§3b): Spearman rho + regret (decision-relevant)
  - ood_benchmark: Mahalanobis AUROC in-dist vs OOD regime
  - uncertainty_validity: error-correlation + OOD/in-dist gap
  - abstention_gain: trusted-subset error vs all-query error
  - horizon_growth_fit: fitted (not asserted) compounding factor
  - sufficiency_probe (§2): within- vs across-group outcome variance
  - stress_matrix: train regime A -> test regime B degradation
  - run_audit: end-to-end report + maturity assignment per MATURITY_GATES
"""

from __future__ import annotations

import math
import random
from typing import Any


MATURITY_GATES = {
    "rho_counterfactual": 0.7,   # L3: ranking preserved under intervention
    "regret_fraction": 0.10,     # L3: regret <= 10% of effect range
    "auroc_mechanistic": 0.75,   # L4: OOD detector separates regimes
    "active_rounds": 3,          # L5: >=3 active rounds beating random
}


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = _ranks(x), _ranks(y)
    n = len(x)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def bootstrap_ci(
    values: list[float], stat: str = "mean", ci: float = 0.95,
    resamples: int = 1000, seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        if stat == "mean":
            stats.append(sum(sample) / n)
        elif stat == "median":
            s = sorted(sample)
            stats.append(s[n // 2])
    stats.sort()
    lo_q = (1.0 - ci) / 2.0
    return (stats[int(lo_q * resamples)], stats[int((1.0 - lo_q) * resamples)])


def arm_empirics(
    nodes: int, arms: list[str], episodes_per_arm: int, seed: int,
    failure_rate: float = 0.0, log_lag_rate: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Ground-truth per-arm outcome rates with Wilson CIs (the P side)."""
    from rlraft.sim.sim import TIMEOUT_ARMS, generate_conditions, simulate_failover
    from rlraft.virtual.trust import wilson_interval

    out: dict[str, dict[str, float]] = {}
    rng = random.Random(seed)
    for arm in arms:
        low, high = TIMEOUT_ARMS[arm]

        class _P:
            def timeout_ms(self, obs, r):  # noqa: ANN001
                return r.uniform(low, high)

        ok = splits = 0
        times: list[float] = []
        for _ in range(episodes_per_arm):
            conds = generate_conditions(nodes, rng, failure_rate=failure_rate,
                                        log_lag_rate=log_lag_rate)
            res = simulate_failover(conds, _P(), rng)
            ok += int(res.success)
            splits += int(res.split_votes > 0)
            times.append(res.total_time_ms)
        lo, hi = wilson_interval(ok, episodes_per_arm)
        out[arm] = {
            "success_rate": ok / episodes_per_arm,
            "success_lo": lo, "success_hi": hi,
            "split_rate": splits / episodes_per_arm,
            "mean_time_ms": sum(times) / len(times),
        }
    return out


def value_consistency(
    predicted: dict[str, float], empirics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """§3a: |E_V - E_P| per arm + MAE + CI-coverage check."""
    errors = {a: abs(predicted[a] - empirics[a]["success_rate"]) for a in predicted}
    covered = sum(
        1 for a in predicted
        if empirics[a]["success_lo"] - 1e-9 <= predicted[a] <= empirics[a]["success_hi"] + 1e-9
    )
    return {
        "per_arm_abs_error": errors,
        "mae": sum(errors.values()) / max(len(errors), 1),
        "max_error": max(errors.values()) if errors else 0.0,
        "ci_coverage": covered / max(len(predicted), 1),
    }


def ranking_consistency(
    predicted: dict[str, float], empirics: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """§3b: does the surrogate rank interventions like reality? + regret."""
    arms = list(predicted)
    pred_order = [predicted[a] for a in arms]
    emp_order = [empirics[a]["success_rate"] for a in arms]
    rho = spearman(pred_order, emp_order)
    best_emp = max(arms, key=lambda a: empirics[a]["success_rate"])
    best_pred = max(arms, key=lambda a: predicted[a])
    regret = empirics[best_emp]["success_rate"] - empirics[best_pred]["success_rate"]
    effect_range = max(emp_order) - min(emp_order)
    return {
        "spearman_rho": rho,
        "regret": regret,
        "effect_range": effect_range,
        "regret_fraction": regret / max(effect_range, 1e-9),
        "best_empirical_arm": best_emp,
        "best_predicted_arm": best_pred,
    }


def uncertainty_validity(
    uncertainties: list[float], abs_errors: list[float],
    in_dist_u: list[float] | None = None, ood_u: list[float] | None = None,
) -> dict[str, float]:
    out = {"error_correlation": pearson(uncertainties, abs_errors)}
    if in_dist_u and ood_u:
        out["mean_in_dist_u"] = sum(in_dist_u) / len(in_dist_u)
        out["mean_ood_u"] = sum(ood_u) / len(ood_u)
        out["ood_gap"] = out["mean_ood_u"] - out["mean_in_dist_u"]
    return out


def abstention_gain(
    abs_errors: list[float], trusted: list[bool],
) -> dict[str, float]:
    all_err = sum(abs_errors) / max(len(abs_errors), 1)
    trusted_errs = [e for e, t in zip(abs_errors, trusted) if t]
    trusted_err = sum(trusted_errs) / max(len(trusted_errs), 1)
    return {
        "error_all": all_err,
        "error_trusted": trusted_err,
        "coverage": sum(trusted) / max(len(trusted), 1),
        "gain": all_err - trusted_err,  # >0 means the gate adds value
    }


def interval_coverage(
    abs_errors: list[float], stds: list[float], k: float = 2.0,
) -> float:
    """Fraction of held-out truths within k ensemble-std of the mean.

    The continuous counterpart to ECE: uncertainty is only useful if it
    bounds error. Reports the raw fraction (no target worship -- compare
    against the k-sigma Gaussian reference ~0.95 only qualitatively).
    """
    n = len(abs_errors)
    if n == 0:
        return 0.0
    return sum(1 for e, s in zip(abs_errors, stds)
               if e <= k * max(s, 1e-12)) / n


def paired_compare(
    a: list[float], b: list[float], resamples: int = 2000, seed: int = 0,
) -> dict[str, float]:
    """Paired A-vs-B comparison: diffs[i] = a[i] - b[i] (negative favors A).

    Returns mean diff with bootstrap CI plus the win rate P(a < b).
    The research-grade replacement for single-number shootouts (e.g.
    learned evictor vs LRU across a workload suite).
    """
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError("paired_compare needs two non-empty equal lists")
    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    lo, hi = bootstrap_ci(diffs, resamples=resamples, seed=seed)
    return {"mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "win_rate_a": sum(1 for d in diffs if d < 0) / n, "n": n}


def horizon_growth_fit(horizons: list[int], errors: list[float]) -> dict[str, float]:
    """Fit error(h) = base * growth**(h-1) in log space. Returns fitted
    growth; the old 1.35 constant is only a prior when data is missing."""
    import math as _m

    if len(horizons) < 2:
        return {"growth": 1.35, "fitted": 0.0}
    xs = [h - 1 for h in horizons]
    ys = [_m.log(max(e, 1e-6)) for e in errors]
    r = pearson([float(x) for x in xs], ys)
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var_x = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(var_x, 1e-9)
    return {"growth": float(_m.exp(max(slope, 0.0))), "log_fit_r": r, "fitted": 1.0}


def sufficiency_probe(
    episodes: list[tuple[list[float], dict[str, float]]], group_dims: int = 5,
) -> dict[str, float]:
    """§2: within-group vs across-group outcome variance under grouping by
    the first group_dims features (rounded). Ratio << 1 supports sufficiency
    of the abstraction; ratio ~1 means A discarded relevant distinctions."""
    groups: dict[tuple, list[float]] = {}
    for feats, outcome in episodes:
        key = tuple(round(f, 1) for f in feats[:group_dims])
        groups.setdefault(key, []).append(outcome["success"])
    all_y = [o["success"] for _, o in episodes]
    mean_all = sum(all_y) / max(len(all_y), 1)
    across = sum((y - mean_all) ** 2 for y in all_y) / max(len(all_y), 1)
    within_num = within_den = 0.0
    for ys in groups.values():
        if len(ys) < 2:
            continue
        m = sum(ys) / len(ys)
        within_num += sum((y - m) ** 2 for y in ys)
        within_den += len(ys)
    within = within_num / max(within_den, 1)
    return {
        "within_var": within, "across_var": across,
        "ratio": within / max(across, 1e-9), "n_groups": len(groups),
    }


def ood_benchmark(
    detector, in_rows: list[list[float]], ood_rows: list[list[float]],
    percentile: float = 95.0,
) -> dict[str, float]:
    from rlraft.virtual.trust import MahalanobisOOD

    in_scores = detector.scores(in_rows)
    ood_scores = detector.scores(ood_rows)
    budget = MahalanobisOOD.budget_from_percentile(in_scores, percentile)
    fpr = sum(1 for s in in_scores if s > budget) / max(len(in_scores), 1)
    tpr = sum(1 for s in ood_scores if s > budget) / max(len(ood_scores), 1)
    return {
        "auroc": MahalanobisOOD.auroc(in_scores, ood_scores),
        "budget_p95": budget,
        "fpr_at_budget": fpr, "tpr_at_budget": tpr,
        "mean_in": sum(in_scores) / max(len(in_scores), 1),
        "mean_ood": sum(ood_scores) / max(len(ood_scores), 1),
    }


def run_audit(
    train_nodes: list[int] | None = None,
    test_nodes: int = 20,
    ood_nodes: int = 100,
    episodes_per_arm: int = 30,
    ensemble_members: int = 3,
    epochs: int = 80,
    seed: int = 7,
    stress_loss: bool = True,
) -> dict[str, Any]:
    """End-to-end rigorous audit. Small by default; scale up for papers."""
    from rlraft.virtual.algorithms import RaftElection
    from rlraft.virtual.base import Intervention
    from rlraft.virtual.surrogate import (
        EnsembleDynamicsSurrogate, TrajectoryDataset, cluster_features,
    )
    from rlraft.virtual.trust import MahalanobisOOD

    train_nodes = train_nodes or [5]
    arms = list(RaftElection.ARMS)
    rng = random.Random(seed)

    # 1. per-episode training data + ensemble surrogate
    train_ds = TrajectoryDataset().collect_episodes(train_nodes, episodes_per_arm, seed)
    train_rows, val_rows = train_ds.train_val_split(0.3, seed)
    ens = EnsembleDynamicsSurrogate(members=ensemble_members)
    ens.train(train_rows, epochs=epochs, seed=seed)

    # 2. validation error + uncertainty validity on held-out episodes
    val_errs, val_u, val_trusted = [], [], []
    for feats, outcome in val_rows:
        p = ens.predict(feats)
        val_errs.append(abs(p["success_prob"] - outcome["success"]))
        val_u.append(p["uncertainty"])
        val_trusted.append(p["uncertainty"] < 0.3)
    mae_val = sum(val_errs) / max(len(val_errs), 1)
    mae_ci = bootstrap_ci(val_errs, resamples=500, seed=seed)
    uval = uncertainty_validity(val_u, val_errs)

    # 3. intervention ranking vs ground truth on test regime
    empirics = arm_empirics(test_nodes, arms, episodes_per_arm, seed + 1)
    state_feats = {
        a: cluster_features(test_nodes, 120.0, 0.02, 0.2, False, i)
        for i, a in enumerate(arms)
    }
    predicted = {a: ens.predict(state_feats[a])["success_prob"] for a in arms}
    vc = value_consistency(predicted, empirics)
    rc = ranking_consistency(predicted, empirics)

    # 4. OOD benchmark: train-regime vs ood-regime features
    ood_ds = TrajectoryDataset().collect_episodes([ood_nodes], max(episodes_per_arm // 3, 5), seed + 2)
    det = MahalanobisOOD()
    det.fit([f for f, _ in train_rows])
    oodb = ood_benchmark(det, [f for f, _ in val_rows], [f for f, _ in ood_ds.episodes])
    ood_u = [ens.predict(f)["uncertainty"] for f, _ in ood_ds.episodes]
    uval["mean_ood_u_stress"] = sum(ood_u) / max(len(ood_u), 1)
    uval["ood_gap_stress"] = uval["mean_ood_u_stress"] - sum(val_u) / max(len(val_u), 1)

    # 5. abstention: would a trust gate on uncertainty help?
    abst = abstention_gain(val_errs, val_trusted)

    # 6. stress: high-loss regime ranking
    stress: dict[str, Any] = {}
    if stress_loss:
        emp_s = arm_empirics(test_nodes, arms, max(episodes_per_arm // 2, 10),
                             seed + 3, failure_rate=0.15, log_lag_rate=0.4)
        stress = {"ranking_under_loss": ranking_consistency(predicted, emp_s),
                  "value_under_loss": value_consistency(predicted, emp_s)}

    # 7. sufficiency probe on training episodes
    suf = sufficiency_probe(train_ds.episodes)

    # 8. maturity assignment per MATURITY_GATES
    l3 = rc["spearman_rho"] >= MATURITY_GATES["rho_counterfactual"] and \
        rc["regret_fraction"] <= MATURITY_GATES["regret_fraction"]
    l4 = l3 and oodb["auroc"] >= MATURITY_GATES["auroc_mechanistic"]
    maturity = "MECHANISTIC" if l4 else "COUNTERFACTUAL" if l3 else \
        "INTERPOLATIVE" if mae_val < 0.3 else "EXEMPLAR"

    return {
        "config": {"train_nodes": train_nodes, "test_nodes": test_nodes,
                   "ood_nodes": ood_nodes, "episodes_per_arm": episodes_per_arm,
                   "members": ensemble_members, "epochs": epochs, "seed": seed},
        "heldout_mae": mae_val, "heldout_mae_ci95": mae_ci,
        "uncertainty_validity": uval,
        "value_consistency": vc, "ranking_consistency": rc,
        "ood_benchmark": oodb, "abstention": abst,
        "stress": stress, "sufficiency": suf, "maturity": maturity,
        "gates": MATURITY_GATES,
    }
